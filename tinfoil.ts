// pi only aliases a few pi-ai specifiers for extensions. "/compat" is one,
// and is a superset of the root entry; deeper paths do not resolve.
import {
	createProvider,
	envApiKeyAuth,
	lazyStream,
	type Model,
	openAICompletionsApi,
	type ProviderStreams,
} from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Box, Text } from "@earendil-works/pi-tui";
// Resolves from this package's node_modules: pi loads extensions with jiti,
// which uses normal Node resolution for anything it does not alias.
import { RouterHTTPClient } from "tinfoil";

/** What a successful verification proves; the SDK re-exports it only as `current`. */
type Verification = NonNullable<RouterHTTPClient["current"]>;

/**
 * Tinfoil provider for the pi coding agent.
 *
 * Attestation happens in-process via the `tinfoil` SDK: it verifies the
 * enclave's attestation document against the Sigstore-signed release, and pins
 * every connection to the attested TLS key so only the verified enclave can
 * read a request. Not a full external verifier; for that use
 * github.com/tinfoilsh/tinfoil-cli.
 *
 * See README.md for setup.
 */

const PROVIDER_ID = "tinfoil";
const ENTRY_TYPE = "tinfoil-report";
const HELP_URL = "https://tinfoil.sh/coding-agents";

const DISCOVER_TIMEOUT_MS = 8000;

// =============================================================================
// Verification state
// =============================================================================

/**
 * The startup verdict. "failed" blocks every request (see the guard).
 *
 * No "stale" state and no refresh TTL: a client only exists once verified, and
 * its `fetch` verifies again on its own when the verification expires or the
 * enclave rotates keys. Renderers read `client.current` on every call, so a
 * recovered rotation still shows up.
 */
type VerifyState =
	| { kind: "verified"; client: RouterHTTPClient }
	| { kind: "failed"; reason: string };

// =============================================================================
// Model discovery
// =============================================================================

interface TinfoilApiModel {
	id?: string;
	name?: string;
	type?: string;
	endpoints?: string[];
	context_window?: number;
	max_tokens?: number;
	reasoning?: boolean;
	multimodal?: boolean;
	tool_calling?: boolean;
	pricing?: {
		inputTokenPricePer1M?: number;
		outputTokenPricePer1M?: number;
		cachedInputTokenPricePer1M?: number;
		requestPrice?: number;
	};
}

/** /v1/models reports no output-token limit; derive a conservative one. */
function deriveMaxTokens(contextWindow: number): number {
	return Math.min(32768, Math.max(4096, Math.floor(contextWindow / 8)));
}

function toModel(raw: TinfoilApiModel, baseUrl: string): Model<"openai-completions"> {
	const contextWindow = raw.context_window ?? 128000;
	return {
		id: raw.id as string,
		name: raw.name ?? (raw.id as string),
		api: "openai-completions",
		provider: PROVIDER_ID,
		baseUrl,
		reasoning: raw.reasoning === true,
		input: raw.multimodal ? ["text", "image"] : ["text"],
		cost: {
			input: raw.pricing?.inputTokenPricePer1M ?? 0,
			output: raw.pricing?.outputTokenPricePer1M ?? 0,
			cacheRead: raw.pricing?.cachedInputTokenPricePer1M ?? 0,
			cacheWrite: 0,
		},
		contextWindow,
		maxTokens: raw.max_tokens ?? deriveMaxTokens(contextWindow),
		// Tinfoil's open-weight backends reject the OpenAI-only `developer` role.
		compat: { supportsDeveloperRole: false },
	};
}

/** A coding agent needs chat plus tool calling; anything else fails in the picker. */
function isUsable(raw: TinfoilApiModel): boolean {
	if (!raw.id) return false;
	if (raw.type && raw.type !== "chat") return false;
	if (raw.endpoints && !raw.endpoints.includes("/v1/chat/completions")) return false;
	if (raw.tool_calling === false) return false;
	return true;
}

/**
 * Used only when live discovery is impossible (verification failed or
 * /v1/models errored). Inevitably stale, but safe: the guard still blocks
 * all requests while unverified. Snapshot of https://inference.tinfoil.sh/v1/models,
 * 2026-08; keep current when shipping releases.
 */
const FALLBACK_CATALOG: TinfoilApiModel[] = [
	{
		id: "kimi-k3",
		name: "Kimi K3",
		type: "chat",
		context_window: 262144,
		reasoning: true,
		multimodal: true,
		tool_calling: true,
		pricing: { inputTokenPricePer1M: 4, outputTokenPricePer1M: 20 },
	},
	{
		id: "deepseek-v4-flash",
		name: "DeepSeek V4 Flash",
		type: "chat",
		context_window: 1048576,
		reasoning: true,
		tool_calling: true,
		pricing: { inputTokenPricePer1M: 0.3, outputTokenPricePer1M: 0.7 },
	},
	{
		id: "glm-5-2",
		name: "GLM-5.2",
		type: "chat",
		context_window: 393216,
		reasoning: true,
		tool_calling: true,
		pricing: { inputTokenPricePer1M: 1.5, outputTokenPricePer1M: 5.25 },
	},
	{
		id: "gpt-oss-120b",
		name: "GPT-OSS 120B",
		type: "chat",
		context_window: 131072,
		reasoning: true,
		tool_calling: true,
		pricing: { inputTokenPricePer1M: 0.15, outputTokenPricePer1M: 0.6 },
	},
	{
		id: "llama3-3-70b",
		name: "Llama 3.3 70B",
		type: "chat",
		context_window: 131072,
		tool_calling: true,
		pricing: { inputTokenPricePer1M: 1.75, outputTokenPricePer1M: 2.75 },
	},
	{
		id: "gemma4-31b",
		name: "Gemma 4 31B",
		type: "chat",
		context_window: 262144,
		reasoning: true,
		multimodal: true,
		tool_calling: true,
		pricing: { inputTokenPricePer1M: 0.4, outputTokenPricePer1M: 1 },
	},
];

interface Discovery {
	models: Model<"openai-completions">[];
	skipped: string[];
	usedFallback: boolean;
	error?: string;
}

/**
 * Discover models through the verified client; the response is OpenAI-shaped.
 * The client's `fetch` refuses any origin but the verified enclave.
 */
async function discoverModels(client: RouterHTTPClient, baseUrl: string): Promise<Discovery> {
	try {
		const response = await client.fetch(new URL("v1/models", client.baseURL), {
			signal: AbortSignal.timeout(DISCOVER_TIMEOUT_MS),
		});
		if (!response.ok) throw new Error(`HTTP ${response.status} from /v1/models`);
		const body = (await response.json()) as { data?: TinfoilApiModel[] };
		const raw = body.data ?? [];
		return {
			models: raw.filter(isUsable).map((entry) => toModel(entry, baseUrl)),
			skipped: raw.filter((entry) => !isUsable(entry)).map((entry) => entry.id ?? "(unnamed)"),
			usedFallback: false,
		};
	} catch (error) {
		return {
			models: FALLBACK_CATALOG.map((entry) => toModel(entry, baseUrl)),
			skipped: [],
			usedFallback: true,
			error: errorMessage(error),
		};
	}
}

// =============================================================================
// Rendering
// =============================================================================

// Plain markers, not emoji: padlock glyphs differ in width across fonts, so
// the footer would jitter when the state flips.
const MARK_OK = "[\u2713]";
const MARK_BAD = "[!]";

const shortHash = (value?: string) => (value ? value.replace(/^sha256:/, "").slice(0, 12) : "unknown");

function isTrusted(state: VerifyState): boolean {
	return state.kind === "verified";
}

function mark(trusted: boolean): string {
	return trusted ? MARK_OK : MARK_BAD;
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

/** Verdict and detail as separate parts so callers can color the marker on its own. */
function summaryParts(state: VerifyState, verification?: Verification): { verdict: string; detail: string } {
	if (state.kind !== "verified") return { verdict: "Tinfoil unverified", detail: "" };
	return { verdict: "Tinfoil verified", detail: shortHash(verification?.codeDigest) };
}

/** Plain one-line summary, marker included. */
function summary(state: VerifyState, verification?: Verification): string {
	const { verdict, detail } = summaryParts(state, verification);
	return [verdict, mark(isTrusted(state)), detail].filter(Boolean).join(" ");
}

/** Payload of a `/tinfoil` transcript entry. */
interface TinfoilReport {
	trusted: boolean;
	summary: string;
	lines: string[];
}

/** A measurement as the verification's JSON form gives it: the platform, then registers as hex. */
type Measurement = ReturnType<Verification["toJSON"]>["codeMeasurement"];

/** A measurement's set registers, one per line, labeled as the verifier names them. */
function measurementLines(label: string, measurement: Measurement): string[] {
	const { platform, ...registers } = measurement;
	return [
		`  ${label.padEnd(17)}${platform}`,
		...Object.entries(registers)
			.filter((pair): pair is [string, string] => Boolean(pair[1]))
			.map(([name, value]) => `    ${name.padEnd(15)}${value}`),
	];
}

function reportLines(
	state: VerifyState,
	verification: Verification | undefined,
	baseUrl: string,
	models: number,
): string[] {
	if (state.kind !== "verified" || !verification) {
		return [
			`Base URL:  ${baseUrl}`,
			`Reason:    ${state.kind === "failed" ? state.reason : "no verification"}`,
			"",
			`Run /tinfoil to retry verification. See ${HELP_URL}`,
		];
	}

	// The JSON form renders registers as hex and drops facts that were not established.
	const facts = verification.toJSON();
	return [
		"Connection",
		`  Base URL:        ${baseUrl}`,
		`  Enclave host:    ${state.client.enclave || "unknown"}`,
		`  Config repo:     ${facts.configRepo || "unknown"}`,
		`  Models:          ${models}`,
		"",
		"Release",
		`  Tag:             ${facts.codeTag || "unknown"}`,
		`  Digest:          ${facts.codeDigest || "unknown"}`,
		...(facts.runtime ? [`  Runtime:         ${facts.runtime.repo}@${facts.runtime.tag} (${shortHash(facts.runtime.digest)})`] : []),
		...(facts.config ? [`  Config:          ${facts.config.name || "unnamed"} (${shortHash(facts.config.digest)})`] : []),
		"",
		"Attested keys",
		`  TLS fingerprint: ${verification.tlsPublicKeyFP ?? "unknown"}`,
		`  HPKE public key: ${verification.hpkePublicKey ?? "unknown"}`,
		"",
		"Measurements",
		...measurementLines("Code:", facts.codeMeasurement),
		...measurementLines("Enclave:", facts.enclaveMeasurement),
		"",
		"Verifier",
		`  Verifier:        ${facts.metadata.verifier.name} ${facts.metadata.verifier.version}`,
		`  Verified at:     ${facts.metadata.verifiedAt}`,
		`  Fresh until:     ${facts.freshnessExpiresAt ?? "no expiry"}`,
	];
}

// =============================================================================
// Extension
// =============================================================================

export default async function (pi: ExtensionAPI) {
	// Fail closed on the extension, not on load: a startup failure (e.g.
	// transient network) must not prevent pi from loading. The guard turns
	// this state into a hard request block; /tinfoil retries it.
	//
	// Default config only: discovers the router and verifies its attestation
	// document against the Sigstore-signed release before the client exists.
	// A retry first re-verifies the router we have, and falls back to fresh
	// discovery, which may pick a different router.
	let client: RouterHTTPClient | undefined;
	const verify = async (): Promise<VerifyState> => {
		if (client) {
			try {
				await client.reverify();
				return { kind: "verified", client };
			} catch {
				void client.close();
				client = undefined;
			}
		}
		try {
			client = await RouterHTTPClient.create();
			return { kind: "verified", client };
		} catch (error) {
			return { kind: "failed", reason: errorMessage(error) };
		}
	};

	let state: VerifyState = await verify();

	/**
	 * When verified, the SDK's resolved enclave URL, so requests land on the
	 * host the attestation covers. When not, a placeholder default: the guard
	 * blocks every request while unverified, so it never reaches the wire.
	 */
	const DEFAULT_API_URL = "https://inference.tinfoil.sh/v1";
	const baseUrlOf = (): string => (state.kind === "verified" ? `${state.client.baseURL}v1` : DEFAULT_API_URL);

	const baseUrl = baseUrlOf();

	/**
	 * Wrap pi's OpenAI-completions adapter so every request rides the verified,
	 * attested channel.
	 *
	 * 1. Fail closed: if verification failed, throw before any byte (key, system
	 *    prompt, tools, user prompt) leaves the machine. Notifications only
	 *    exist in interactive mode, so the request path is the only place that
	 *    covers print/JSON runs too.
	 * 2. Inject the verified client's `fetch`: pi-ai passes `options.fetch` to
	 *    the OpenAI client, so this one hook covers every request. The SDK pins
	 *    each connection to the attested key, refuses origins other than the
	 *    verified enclave, and re-verifies on its own when the server rotates keys.
	 */
	const guard = (streams: ProviderStreams): ProviderStreams => {
		const requireVerified = async () => {
			if (state.kind !== "verified") {
				throw new Error(
					`Tinfoil: refusing to send this request. Enclave verification failed: ${state.reason} ` +
						`Run /tinfoil to retry, or see ${HELP_URL}`,
				);
			}
			return state.client.fetch;
		};
		return {
			stream: (model, context, options) =>
				lazyStream(model, async () => {
					const fetch = await requireVerified();
					// Models keep the baseUrl from load time; a /tinfoil re-verify may
					// have moved to another enclave, so pin to the verified one.
					return streams.stream({ ...model, baseUrl: baseUrlOf() }, context, { ...options, fetch });
				}),
			streamSimple: (model, context, options) =>
				lazyStream(model, async () => {
					const fetch = await requireVerified();
					return streams.streamSimple({ ...model, baseUrl: baseUrlOf() }, context, { ...options, fetch });
				}),
		};
	};

	// Discovery runs only when verified; when unverified there is no trusted
	// channel, so register the fallback catalog. The picker stays usable and
	// the guard still blocks requests.
	let discovery: Discovery =
		state.kind === "verified"
			? await discoverModels(state.client, baseUrl)
			: {
					models: FALLBACK_CATALOG.map((entry) => toModel(entry, baseUrl)),
					skipped: [],
					usedFallback: true,
					error: `skipped (verification failed): ${state.reason}`,
				};

	pi.registerProvider(
		createProvider({
			id: PROVIDER_ID,
			name: "Tinfoil",
			baseUrl,
			auth: { apiKey: envApiKeyAuth("Tinfoil API key", ["TINFOIL_API_KEY"]) },
			// Static list, discovered per extension load. `fetchModels` would be
			// dead code: `pi update --models` runs without extensions loaded.
			models: discovery.models,
			api: guard(openAICompletionsApi()),
		}),
	);

	/** Both session_start and command contexts expose these members. */
	type StatusContext = {
		model?: { provider?: string };
		ui: {
			setStatus(key: string, text: string | undefined): void;
			notify(message: string, type?: "info" | "warning" | "error"): void;
			theme?: { fg(token: string, text: string): string };
		};
	};

	const usesTinfoil = (ctx: StatusContext) => ctx.model?.provider === PROVIDER_ID;

	// Fresh from the SDK at render time, so a rotation it recovered from on
	// its own still shows up without us tracking it.
	const verificationOf = (): Verification | undefined => (state.kind === "verified" ? state.client.current : undefined);

	// The footer belongs to the active model; a Tinfoil status next to another
	// provider's model would claim a guarantee that does not apply.
	const showStatus = (ctx: StatusContext, active = usesTinfoil(ctx)) => {
		if (!active) {
			ctx.ui.setStatus(PROVIDER_ID, undefined);
			return;
		}
		const theme = ctx.ui.theme;
		if (!theme) {
			ctx.ui.setStatus(PROVIDER_ID, summary(state, verificationOf()));
			return;
		}
		const trusted = isTrusted(state);
		const { verdict, detail } = summaryParts(state, verificationOf());
		const parts = [
			theme.fg("dim", verdict),
			theme.fg(trusted ? "success" : "error", mark(trusted)),
			detail ? theme.fg("dim", detail) : "",
		];
		ctx.ui.setStatus(PROVIDER_ID, parts.filter(Boolean).join(" "));
	};

	/** The warning that the current state deserves, or undefined when all is well. */
	const warning = (): { message: string; level: "warning" | "error" } | undefined => {
		if (state.kind === "failed") {
			return {
				message:
					`Tinfoil: enclave verification failed: ${state.reason} ` +
					`Requests are blocked. Run /tinfoil for the full report. See ${HELP_URL}`,
				level: "error",
			};
		}
		return undefined;
	};

	// A transcript entry rather than a notification: it renders in muted theme
	// colors and stays out of the LLM context.
	pi.registerEntryRenderer<TinfoilReport>(ENTRY_TYPE, (entry, { expanded }, theme) => {
		const data = entry.data;
		const box = new Box(1, 0);
		if (!data) {
			box.addChild(new Text(theme.fg("dim", `${MARK_BAD} Tinfoil: no report`), 0, 0));
			return box;
		}

		const token = data.trusted ? "success" : "error";
		box.addChild(new Text(theme.fg(token, data.summary), 0, 0));
		for (const line of expanded ? data.lines : []) {
			box.addChild(new Text(theme.fg("dim", line), 0, 0));
		}
		if (!expanded) box.addChild(new Text(theme.fg("dim", "  Expand for the full verification report."), 0, 0));
		return box;
	});

	pi.registerCommand(PROVIDER_ID, {
		description: "Verify the Tinfoil enclave again and show the report",
		handler: async (_args, ctx) => {
			// Verifies again even if the cached verification is still valid. Also
			// the recovery path for a startup failure.
			state = await verify();

			// A fresh verification may resolve a different enclave or catalog.
			if (state.kind === "verified") {
				discovery = await discoverModels(state.client, baseUrlOf());
			}

			showStatus(ctx);
			pi.appendEntry<TinfoilReport>(ENTRY_TYPE, {
				trusted: isTrusted(state),
				summary: summary(state, verificationOf()),
				lines: reportLines(state, verificationOf(), baseUrlOf(), discovery.models.length),
			});
		},
	});

	// Warn at model selection: that is when the guarantee starts to matter.
	pi.on("model_select", async (event, ctx) => {
		if (event.model?.provider !== PROVIDER_ID) {
			ctx.ui.setStatus(PROVIDER_ID, undefined);
			return;
		}
		showStatus(ctx, true);
		const problem = warning();
		if (problem) ctx.ui.notify(problem.message, problem.level);
	});

	pi.on("session_start", async (_event, ctx) => {
		showStatus(ctx);

		// One message per session; a stack of warnings trains users to ignore them.
		const problem = warning();

		// Stay quiet for users of other providers. Exception: a provider with no
		// models can never be selected, so silence would hide the cause forever.
		if (!usesTinfoil(ctx) && discovery.models.length > 0) return;

		if (problem) {
			ctx.ui.notify(problem.message, problem.level);
		} else if (discovery.usedFallback) {
			ctx.ui.notify(
				`Tinfoil: model discovery failed (${discovery.error}), so the built-in fallback catalog is in use.`,
				"warning",
			);
		} else if (!discovery.models.length) {
			ctx.ui.notify("Tinfoil: the enclave served no usable chat models.", "warning");
		}
	});
}
