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
import { SecureClient, type VerificationDocument } from "tinfoil";

/**
 * Tinfoil provider for the pi coding agent.
 *
 * Attestation and encryption happen in-process via the `tinfoil` SDK: it
 * verifies the enclave's SEV-SNP attestation and Sigstore-signed code digest,
 * and encrypts every request body end-to-end with HPKE. Not a full external
 * verifier (no independent AMD signature-chain check); for that use
 * github.com/tinfoilsh/tinfoil-cli.
 *
 * See README.md for setup.
 */

const PROVIDER_ID = "tinfoil";
const ENTRY_TYPE = "tinfoil-report";
const HELP_URL = "https://tinfoil.sh/coding-agents";

const DISCOVER_TIMEOUT_MS = 8000;

// The document layout this extension was written against. Drift is a
// warning, not a failure: the attestation behind the fields is still checked.
const KNOWN_SCHEMA_VERSION = 1;

// =============================================================================
// Verification state
// =============================================================================

/**
 * The startup verdict. "failed" blocks every request (see the guard).
 *
 * No "stale" state and no refresh TTL: verification is bound to the live
 * connection and `SecureClient.fetch` re-attests on its own when the enclave
 * rotates keys. Renderers read the document fresh on every call, so a
 * recovered rotation still shows up.
 */
type VerifyState =
	| { kind: "verified" }
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
			cacheRead: 0,
			cacheWrite: 0,
		},
		contextWindow,
		maxTokens: raw.max_tokens ?? deriveMaxTokens(contextWindow),
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
 * Discover models through the encrypted, verified channel; the response is
 * OpenAI-shaped. `SecureClient.fetch` resolves the relative URL against the
 * verified enclave and refuses any other origin.
 */
async function discoverModels(secureFetch: typeof fetch, baseUrl: string): Promise<Discovery> {
	try {
		const response = await secureFetch("/v1/models", {
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
			error: error instanceof Error ? error.message : String(error),
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
function summaryParts(state: VerifyState, document?: VerificationDocument): { verdict: string; detail: string } {
	if (state.kind !== "verified") return { verdict: "Tinfoil unverified", detail: "" };
	return { verdict: "Tinfoil verified", detail: shortHash(document?.releaseDigest) };
}

/** Plain one-line summary, marker included. */
function summary(state: VerifyState, document?: VerificationDocument): string {
	const { verdict, detail } = summaryParts(state, document);
	return [verdict, mark(isTrusted(state)), detail].filter(Boolean).join(" ");
}

/** Payload of a `/tinfoil` transcript entry. */
interface TinfoilReport {
	trusted: boolean;
	summary: string;
	lines: string[];
}

/** One step line, in the order the verifier performs the steps. */
function stepLines(document: VerificationDocument): string[] {
	const steps = document.steps;
	if (!steps) return [];
	const entries: Array<[string, { status?: string; error?: string } | undefined]> = [
		["Fetch digest", steps.fetchDigest],
		["Verify code", steps.verifyCode],
		["Verify enclave", steps.verifyEnclave],
		["Compare measurements", steps.compareMeasurements],
		["Verify certificate", steps.verifyCertificate],
	];
	return entries
		.filter((pair): pair is [string, { status?: string; error?: string }] => pair[1] !== undefined)
		.map(([name, step]) => `  ${name.padEnd(22)}${step.status ?? "unknown"}${step.error ? `: ${step.error}` : ""}`);
}

function reportLines(
	state: VerifyState,
	document: VerificationDocument | undefined,
	baseUrl: string,
	models: number,
): string[] {
	if (state.kind !== "verified" || !document) {
		return [
			`Base URL:  ${baseUrl}`,
			`Reason:    ${state.kind === "failed" ? state.reason : "no verification document"}`,
			"",
			`Run /tinfoil to retry verification. See ${HELP_URL}`,
		];
	}

	const enclave = document.enclaveMeasurement ?? {};
	return [
		"Connection",
		`  Base URL:        ${baseUrl}`,
		`  Enclave host:    ${document.enclaveHost || "unknown"}`,
		`  Router endpoint: ${document.selectedRouterEndpoint || "unknown"}`,
		`  Config repo:     ${document.configRepo || "unknown"}`,
		`  Models:          ${models}`,
		"",
		"Release",
		`  Tag:             ${document.releaseTag ?? "unknown"}`,
		`  Digest:          ${document.releaseDigest || "unknown"}`,
		`  Code print:      ${document.codeFingerprint || "unknown"}`,
		`  Enclave print:   ${document.enclaveFingerprint || "unknown"}`,
		"",
		"Attested keys",
		`  TLS public key:  ${document.tlsPublicKey || "unknown"}`,
		`  TLS fingerprint: ${enclave.tlsPublicKeyFingerprint ?? "unknown"}`,
		`  HPKE public key: ${document.hpkePublicKey || enclave.hpkePublicKey || "unknown"}`,
		"",
		"Measurements",
		`  Code:            ${document.codeMeasurement?.type || "unknown"}`,
		...(document.codeMeasurement?.registers ?? []).map((register) => `                   ${register}`),
		`  Enclave:         ${enclave.measurement?.type ?? "unknown"}`,
		...(enclave.measurement?.registers ?? []).map((register) => `                   ${register}`),
		"",
		"Verification steps",
		...stepLines(document),
		"",
		"Verifier",
		`  Verifier:        ${document.verifier?.name ?? "unknown"} ${document.verifier?.version ?? ""}`.trimEnd(),
		`  Verified:        ${document.securityVerified === true ? "yes" : "(SDK did not mark this document verified)"}`,
		`  Verified at:     ${document.verifiedAt ?? "unknown"}`,
	];
}

// =============================================================================
// Extension
// =============================================================================

export default async function (pi: ExtensionAPI) {
	// Default config only: resolves the router, verifies SEV-SNP against the
	// Sigstore-signed release digest, sets up HPKE body encryption.
	const secureClient = new SecureClient();

	// Fail closed on the extension, not on load: a startup failure (e.g.
	// transient network) must not prevent pi from loading. The guard turns
	// this state into a hard request block; /tinfoil retries it.
	const verify = async (): Promise<VerifyState> => {
		try {
			await secureClient.ready();
			return { kind: "verified" };
		} catch (error) {
			return { kind: "failed", reason: errorMessage(error) };
		}
	};

	let state: VerifyState = await verify();

	// Warn (do not fail) if the document layout drifts ahead of this extension.
	const schemaNote = (): string | undefined => {
		if (state.kind !== "verified") return undefined;
		const version = secureClient.getVerificationDocument().schemaVersion;
		return version === KNOWN_SCHEMA_VERSION
			? undefined
			: `the SDK reports document schema ${version ?? "(none)"}, and this extension knows ${KNOWN_SCHEMA_VERSION}. ` +
				`Some fields in /tinfoil may read as "unknown". Update the extension.`;
	};

	/**
	 * When verified, the SDK's resolved enclave URL, so requests land on the
	 * host the attestation covers. When not, a placeholder default: the guard
	 * blocks every request while unverified, so it never reaches the wire.
	 */
	const DEFAULT_API_URL = "https://inference.tinfoil.sh/v1";
	const baseUrlOf = (): string =>
		(state.kind === "verified" ? secureClient.getBaseURL() : undefined)?.replace(/\/+$/, "") ?? DEFAULT_API_URL;

	const baseUrl = baseUrlOf();

	/**
	 * Wrap pi's OpenAI-completions adapter so every request rides the verified,
	 * HPKE-encrypted channel.
	 *
	 * 1. Fail closed: if verification failed, throw before any byte (key, system
	 *    prompt, tools, user prompt) leaves the machine. Notifications only
	 *    exist in interactive mode, so the request path is the only place that
	 *    covers print/JSON runs too.
	 * 2. Inject `secureClient.fetch`: pi-ai passes `options.fetch` to the OpenAI
	 *    client, so this one hook covers every request. The SDK seals each body
	 *    to the attested HPKE key, refuses origins other than the verified
	 *    enclave, and re-attests on its own when the server rotates keys.
	 */
	const guard = (streams: ProviderStreams): ProviderStreams => {
		const ensureVerified = async () => {
			if (state.kind !== "verified") {
				throw new Error(
					`Tinfoil: refusing to send this request. Enclave verification failed: ${state.reason} ` +
						`Run /tinfoil to retry, or see ${HELP_URL}`,
				);
			}
		};
		return {
			stream: (model, context, options) =>
				lazyStream(model, async () => {
					await ensureVerified();
					return streams.stream(model, context, { ...options, fetch: secureClient.fetch });
				}),
			streamSimple: (model, context, options) =>
				lazyStream(model, async () => {
					await ensureVerified();
					return streams.streamSimple(model, context, { ...options, fetch: secureClient.fetch });
				}),
		};
	};

	// Discovery runs only when verified; when unverified there is no trusted
	// channel, so register the fallback catalog. The picker stays usable and
	// the guard still blocks requests.
	let discovery: Discovery =
		state.kind === "verified"
			? await discoverModels(secureClient.fetch, baseUrl)
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
	const documentOf = (): VerificationDocument | undefined =>
		state.kind === "verified" ? secureClient.getVerificationDocument() : undefined;

	// The footer belongs to the active model; a Tinfoil status next to another
	// provider's model would claim a guarantee that does not apply.
	const showStatus = (ctx: StatusContext, active = usesTinfoil(ctx)) => {
		if (!active) {
			ctx.ui.setStatus(PROVIDER_ID, undefined);
			return;
		}
		const theme = ctx.ui.theme;
		if (!theme) {
			ctx.ui.setStatus(PROVIDER_ID, summary(state, documentOf()));
			return;
		}
		const trusted = isTrusted(state);
		const { verdict, detail } = summaryParts(state, documentOf());
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
		const note = schemaNote();
		if (note) return { message: `Tinfoil: ${note}`, level: "warning" };
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
		if (!expanded) box.addChild(new Text(theme.fg("dim", "  Expand for the full verification document."), 0, 0));
		return box;
	});

	pi.registerCommand(PROVIDER_ID, {
		description: "Show the Tinfoil enclave verification document",
		handler: async (_args, ctx) => {
			// reset() drops the cached attestation so ready() re-checks everything
			// from scratch. Also the recovery path for a startup failure.
			secureClient.reset();
			state = await verify();

			// A fresh verification may resolve a different enclave or catalog.
			if (state.kind === "verified") {
				discovery = await discoverModels(secureClient.fetch, baseUrlOf());
			}

			showStatus(ctx);
			pi.appendEntry<TinfoilReport>(ENTRY_TYPE, {
				trusted: isTrusted(state),
				summary: summary(state, documentOf()),
				lines: reportLines(state, documentOf(), baseUrlOf(), discovery.models.length),
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
