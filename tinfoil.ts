import { spawn } from "node:child_process";
// pi aliases only four pi-ai specifiers for extensions. "/compat" is one of
// them, and it is a superset of the root entry. Deeper paths such as
// "/api/openai-completions.lazy" do not resolve inside pi.
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

/**
 * Tinfoil provider for the pi coding agent.
 *
 * Points pi at the local Tinfoil Proxy, which verifies the upstream enclave's
 * attestation and pins the attested key. Requests never go straight to the
 * hosted endpoint: that would give audit-time verification only, not the
 * connection-time guarantee this extension exists to preserve.
 *
 * See README.md for setup.
 */

const PROVIDER_ID = "tinfoil";
const ENTRY_TYPE = "tinfoil-report";
const DEFAULT_PORT = 3301;
const DEFAULT_BASE_URL = `http://127.0.0.1:${DEFAULT_PORT}/v1`;
const PROXY_BIN = "tinfoil-proxy";
const HELP_URL = "https://tinfoil.sh/coding-agents";

const PROBE_TIMEOUT_MS = 3000;
const DISCOVER_TIMEOUT_MS = 8000;
const STARTUP_TIMEOUT_MS = 45000;
const STARTUP_POLL_MS = 250;
/** How long a probe result may guard requests before it is re-taken. */
const STATE_TTL_MS = 30000;

/** The document layout this extension was written against. */
const KNOWN_SCHEMA_VERSION = 1;

// =============================================================================
// Verification document
// =============================================================================

/** Subset of the proxy's /verification-document response that we render. */
interface VerificationDocument {
	schemaVersion?: number;
	configRepo?: string;
	enclaveHost?: string;
	releaseTag?: string;
	releaseDigest?: string;
	codeFingerprint?: string;
	enclaveFingerprint?: string;
	tlsPublicKey?: string;
	hpkePublicKey?: string;
	selectedRouterEndpoint?: string;
	securityVerified?: boolean;
	verifiedAt?: string;
	verifier?: { name?: string; version?: string };
	codeMeasurement?: { type?: string; registers?: string[] };
	enclaveMeasurement?: {
		measurement?: { type?: string; registers?: string[] };
		tlsPublicKeyFingerprint?: string;
		hpkePublicKey?: string;
	};
	hardwareMeasurement?: Record<string, unknown>;
	runtime?: {
		instanceId?: string;
		listener?: string;
		software?: { name?: string; version?: string };
	};
}

type ProxyState =
	/** A proxy answered with a document that passed validation. */
	| { kind: "verified"; document: VerificationDocument; note?: string }
	/**
	 * Something answers on the port, but its answer does not prove verification.
	 *
	 * `evidence` separates "we were told something is wrong" from "we could not
	 * ask". Only the first justifies blocking requests. A proxy too old to serve
	 * the document still attests exactly as it always did, so treating silence
	 * as guilt would break working setups while stopping no attacker.
	 */
	| { kind: "unverified"; reason: string; evidence: "bad-document" | "no-endpoint" }
	/** Nothing answers on the port. */
	| { kind: "absent"; reason: string };

/**
 * Reject a document that does not positively state a successful verification.
 *
 * A status indicator must fail closed. A missing field is not a pass, so every
 * check here demands the affirmative value, never the absence of a negative.
 *
 * This does not authenticate the proxy, and it cannot: the document is
 * unsigned, so any local process could replay a genuine one. Trust rests on
 * the assumption that a loopback port belongs to the program that claims it.
 * These checks catch an empty or partial answer, and schema drift.
 */
function validate(document: VerificationDocument): string[] {
	const problems: string[] = [];
	if (document.securityVerified !== true) problems.push("securityVerified is not true");
	if (!document.releaseDigest) problems.push("releaseDigest is missing");
	if (!document.tlsPublicKey) problems.push("tlsPublicKey is missing");
	return problems;
}

function originOf(baseUrl: string): string {
	return new URL(baseUrl).origin;
}

/**
 * Ask the proxy for its verification document.
 *
 * A 404 means the port is held by something that is not a current Tinfoil
 * Proxy: either an outdated proxy, or an unrelated process. Both are reported
 * as "unverified", because pi must not present either one as attested.
 */
async function probeProxy(baseUrl: string, timeoutMs = PROBE_TIMEOUT_MS): Promise<ProxyState> {
	const url = `${originOf(baseUrl)}/verification-document`;
	let response: Response;
	try {
		response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
	} catch (error) {
		return { kind: "absent", reason: error instanceof Error ? error.message : String(error) };
	}
	if (response.status === 404) {
		return {
			kind: "unverified",
			reason: `${url} returned 404. The proxy is outdated, or another program holds the port.`,
			evidence: "no-endpoint",
		};
	}
	if (!response.ok) {
		return { kind: "unverified", reason: `${url} returned HTTP ${response.status}.`, evidence: "no-endpoint" };
	}
	let document: VerificationDocument;
	try {
		document = (await response.json()) as VerificationDocument;
	} catch (error) {
		return {
			kind: "unverified",
			reason: `${url} returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
			evidence: "bad-document",
		};
	}

	const problems = validate(document);
	if (problems.length) {
		return {
			kind: "unverified",
			reason: `${url} did not prove verification: ${problems.join(", ")}.`,
			evidence: "bad-document",
		};
	}

	// Layout drift is worth saying out loud, but it is not a failed
	// verification. Failing closed here would break every user on the day the
	// proxy ships a new schema.
	const version = document.schemaVersion;
	const note =
		version === KNOWN_SCHEMA_VERSION
			? undefined
			: `the proxy reports document schema ${version ?? "(none)"}, and this extension knows ${KNOWN_SCHEMA_VERSION}. ` +
				`Some fields may read as "unknown". Update the extension.`;

	return { kind: "verified", document, note };
}

// =============================================================================
// Proxy autostart
// =============================================================================

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Spawn the proxy detached, so it outlives this pi session and every later
 * session adopts it. We never kill it. The proxy binds a fixed port, so the
 * operating system already guarantees a single instance: a duplicate exits
 * with EADDRINUSE instead of racing.
 */
async function startProxy(baseUrl: string): Promise<{ started: boolean; reason?: string }> {
	const port = new URL(baseUrl).port || String(DEFAULT_PORT);

	let child: ReturnType<typeof spawn>;
	try {
		child = spawn(PROXY_BIN, ["--port", port], { detached: true, stdio: "ignore" });
	} catch (error) {
		return { started: false, reason: error instanceof Error ? error.message : String(error) };
	}

	const spawnFailure = new Promise<string | undefined>((resolve) => {
		child.once("error", (error: NodeJS.ErrnoException) => {
			resolve(error.code === "ENOENT" ? `${PROXY_BIN} is not on PATH.` : error.message);
		});
		child.once("exit", (code) => resolve(`${PROXY_BIN} exited with code ${code}.`));
	});
	child.unref();

	const deadline = Date.now() + STARTUP_TIMEOUT_MS;
	while (Date.now() < deadline) {
		const failure = await Promise.race([spawnFailure, sleep(STARTUP_POLL_MS).then(() => undefined)]);
		if (failure) return { started: false, reason: failure };
		if ((await probeProxy(baseUrl, 1000)).kind !== "absent") return { started: true };
	}
	return { started: false, reason: `${PROXY_BIN} did not become ready in ${STARTUP_TIMEOUT_MS / 1000}s.` };
}

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

/**
 * The API reports no output-token limit, so derive a conservative one from the
 * context window. Replace this once /v1/models exposes the real value.
 */
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

/**
 * A coding agent needs chat plus tool calling. Anything else (embeddings,
 * audio, or a chat model without tools) would appear in the picker and then
 * fail in a way that is hard to read.
 */
function isUsable(raw: TinfoilApiModel): boolean {
	if (!raw.id) return false;
	if (raw.type && raw.type !== "chat") return false;
	if (raw.endpoints && !raw.endpoints.includes("/v1/chat/completions")) return false;
	if (raw.tool_calling === false) return false;
	return true;
}

interface Discovery {
	models: Model<"openai-completions">[];
	skipped: string[];
	error?: string;
}

async function discoverModels(baseUrl: string, signal?: AbortSignal): Promise<Discovery> {
	try {
		const response = await fetch(`${baseUrl}/models`, {
			signal: signal ?? AbortSignal.timeout(DISCOVER_TIMEOUT_MS),
		});
		if (!response.ok) throw new Error(`HTTP ${response.status} from ${baseUrl}/models`);
		const body = (await response.json()) as { data?: TinfoilApiModel[] };
		const raw = body.data ?? [];
		return {
			models: raw.filter(isUsable).map((entry) => toModel(entry, baseUrl)),
			skipped: raw.filter((entry) => !isUsable(entry)).map((entry) => entry.id ?? "(unnamed)"),
		};
	} catch (error) {
		return {
			models: [],
			skipped: [],
			error: error instanceof Error ? error.message : String(error),
		};
	}
}

// =============================================================================
// Rendering
// =============================================================================

/**
 * Plain markers, not emoji. Emoji padlocks are double-width, and the open and
 * closed glyphs differ in width across fonts, so the footer jitters when the
 * state flips. Color carries the meaning; the marker carries it again for
 * users who cannot rely on color.
 */
const MARK_OK = "[\u2713]";
const MARK_BAD = "[!]";

const shortHash = (value?: string) => (value ? value.replace(/^sha256:/, "").slice(0, 12) : "unknown");

function isTrusted(state: ProxyState): boolean {
	return state.kind === "verified";
}

function mark(trusted: boolean): string {
	return trusted ? MARK_OK : MARK_BAD;
}

/**
 * The summary in two parts, so callers can put the marker between them and
 * color each part on its own. The footer dims the words and colors only the
 * marker; the transcript colors the whole line.
 */
function summaryParts(state: ProxyState): { verdict: string; detail: string } {
	if (state.kind !== "verified") return { verdict: "Tinfoil unverified", detail: "" };
	return { verdict: "Tinfoil verified", detail: shortHash(state.document.releaseDigest) };
}

/** Plain one-line summary, marker included. */
function summary(state: ProxyState): string {
	const { verdict, detail } = summaryParts(state);
	return [verdict, mark(isTrusted(state)), detail].filter(Boolean).join(" ");
}

/** Payload of a `/tinfoil` transcript entry. */
interface TinfoilReport {
	trusted: boolean;
	summary: string;
	lines: string[];
}

function reportLines(state: ProxyState, baseUrl: string, models: number): string[] {
	if (state.kind !== "verified") {
		return [
			`Base URL:  ${baseUrl}`,
			`Reason:    ${state.reason}`,
			"",
			state.kind === "absent"
				? `Start the proxy, then run /reload. See ${HELP_URL}`
				: `Update the proxy (tinfoil-proxy --version), then run /reload. See ${HELP_URL}`,
		];
	}

	const document = state.document;
	const enclave = document.enclaveMeasurement ?? {};
	return [
		"Connection",
		`  Base URL:        ${baseUrl}`,
		`  Enclave host:    ${document.enclaveHost ?? "unknown"}`,
		`  Router endpoint: ${document.selectedRouterEndpoint ?? "unknown"}`,
		`  Config repo:     ${document.configRepo ?? "unknown"}`,
		`  Models:          ${models}`,
		"",
		"Release",
		`  Tag:             ${document.releaseTag ?? "unknown"}`,
		`  Digest:          ${document.releaseDigest ?? "unknown"}`,
		`  Code print:      ${document.codeFingerprint ?? "unknown"}`,
		`  Enclave print:   ${document.enclaveFingerprint ?? "unknown"}`,
		"",
		"Attested keys",
		`  TLS public key:  ${document.tlsPublicKey ?? "unknown"}`,
		`  TLS fingerprint: ${enclave.tlsPublicKeyFingerprint ?? "unknown"}`,
		`  HPKE public key: ${document.hpkePublicKey ?? enclave.hpkePublicKey ?? "unknown"}`,
		"",
		"Measurements",
		`  Code:            ${document.codeMeasurement?.type ?? "unknown"}`,
		...(document.codeMeasurement?.registers ?? []).map((register) => `                   ${register}`),
		`  Enclave:         ${enclave.measurement?.type ?? "unknown"}`,
		...(enclave.measurement?.registers ?? []).map((register) => `                   ${register}`),
		"",
		"Verifier",
		`  Verifier:        ${document.verifier?.name ?? "unknown"} ${document.verifier?.version ?? ""}`.trimEnd(),
		`  Proxy:           ${document.runtime?.software?.name ?? "unknown"} ${document.runtime?.software?.version ?? ""}`.trimEnd(),
		`  Instance:        ${document.runtime?.instanceId ?? "unknown"}`,
		`  Verified at:     ${document.verifiedAt ?? "unknown"}`,
	];
}

// =============================================================================
// Extension
// =============================================================================

export default async function (pi: ExtensionAPI) {
	const baseUrl = process.env.TINFOIL_BASE_URL || DEFAULT_BASE_URL;
	const isLoopback = /^https?:\/\/(127\.0\.0\.1|localhost|\[::1\])(:\d+)?(\/|$)/.test(baseUrl);
	const autostart = process.env.TINFOIL_AUTOSTART !== "0";

	let state = await probeProxy(baseUrl);
	let probedAt = Date.now();
	let autostartNote: string | undefined;

	if (state.kind === "absent" && autostart && isLoopback) {
		const result = await startProxy(baseUrl);
		if (result.started) {
			state = await probeProxy(baseUrl);
			probedAt = Date.now();
			autostartNote = `started ${PROXY_BIN} on ${baseUrl}.`;
		} else {
			autostartNote = result.reason;
		}
	}

	/** Re-probe when the cached result is older than the TTL, or when forced. */
	const currentState = async (force = false): Promise<ProxyState> => {
		if (force || Date.now() - probedAt > STATE_TTL_MS) {
			state = await probeProxy(baseUrl);
			probedAt = Date.now();
		}
		return state;
	};

	/**
	 * Refuse a request before any of it leaves the machine.
	 *
	 * The UI cannot carry this decision: notifications exist only in interactive
	 * mode, so print and JSON runs would send the key, system prompt, tools, and
	 * user prompt with no signal at all. The request path is the only place that
	 * covers every mode.
	 *
	 * The probe behind the gate is cached, because verification state changes
	 * only when the enclave rotates, and a rotation cannot produce a bad verdict:
	 * the proxy either re-verifies or fails the request itself.
	 */
	const guard = (streams: ProviderStreams): ProviderStreams => {
		const check = async () => {
			if (!isLoopback) {
				throw new Error(
					`Tinfoil: refusing to send this request. "${baseUrl}" is not the local proxy, so ` +
						`nothing verifies the enclave at connection time. Unset TINFOIL_BASE_URL.`,
				);
			}
			const now = await currentState();
			if (now.kind === "unverified" && now.evidence === "bad-document") {
				throw new Error(
					`Tinfoil: refusing to send this request. ${now.reason} Run /tinfoil for detail, ` +
						`or see ${HELP_URL}`,
				);
			}
		};
		return {
			stream: (model, context, options) =>
				lazyStream(model, async () => {
					await check();
					return streams.stream(model, context, options);
				}),
			streamSimple: (model, context, options) =>
				lazyStream(model, async () => {
					await check();
					return streams.streamSimple(model, context, options);
				}),
		};
	};

	const discovery = await discoverModels(baseUrl);

	pi.registerProvider(
		createProvider({
			id: PROVIDER_ID,
			name: "Tinfoil",
			baseUrl,
			auth: { apiKey: envApiKeyAuth("Tinfoil API key", ["TINFOIL_API_KEY"]) },
			// Discovered above, on every extension load, so each pi start gets the
			// list the proxy serves right now. `fetchModels` would be dead code:
			// `pi update --models` builds a runtime without extensions, so an
			// extension provider never takes part in a catalog refresh. See TODO.md.
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

	/**
	 * The footer belongs to the active model. A Tinfoil status next to an
	 * Anthropic model would claim a guarantee that does not apply to the
	 * request in flight.
	 */
	const showStatus = (ctx: StatusContext, active = usesTinfoil(ctx)) => {
		if (!active) {
			ctx.ui.setStatus(PROVIDER_ID, undefined);
			return;
		}
		const theme = ctx.ui.theme;
		if (!theme) {
			ctx.ui.setStatus(PROVIDER_ID, summary(state));
			return;
		}
		const trusted = isTrusted(state);
		const { verdict, detail } = summaryParts(state);
		const parts = [
			theme.fg("dim", verdict),
			theme.fg(trusted ? "success" : "error", mark(trusted)),
			detail ? theme.fg("dim", detail) : "",
		];
		ctx.ui.setStatus(PROVIDER_ID, parts.filter(Boolean).join(" "));
	};

	const refresh = async (ctx: StatusContext, active?: boolean) => {
		await currentState(true);
		showStatus(ctx, active);
	};

	/** The warning that the current state deserves, or undefined when all is well. */
	const warning = (): { message: string; level: "warning" | "error" } | undefined => {
		if (!isLoopback) {
			return {
				message:
					`Tinfoil: base URL "${baseUrl}" is not the local proxy. Requests bypass ` +
					`connection-time attestation, so the privacy guarantee is NOT verified. ` +
					`Unset TINFOIL_BASE_URL.`,
				level: "warning",
			};
		}
		if (state.kind === "absent") {
			const why = autostartNote ? ` Autostart failed: ${autostartNote}` : "";
			return { message: `Tinfoil: no proxy at ${baseUrl}.${why} See ${HELP_URL}`, level: "warning" };
		}
		if (state.kind === "unverified") {
			return { message: `Tinfoil: ${state.reason} Run /tinfoil for detail. See ${HELP_URL}`, level: "warning" };
		}
		if (state.note) return { message: `Tinfoil: ${state.note}`, level: "warning" };
		return undefined;
	};

	// A transcript entry, not a notification: it renders in muted theme colors,
	// so it never reads as agent output, and it stays out of the LLM context.
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
			await refresh(ctx);
			pi.appendEntry<TinfoilReport>(ENTRY_TYPE, {
				trusted: isTrusted(state),
				summary: summary(state),
				lines: reportLines(state, baseUrl, discovery.models.length),
			});
		},
	});

	// Warn when the user picks a Tinfoil model, because that is the moment the
	// guarantee starts to matter.
	pi.on("model_select", async (event, ctx) => {
		if (event.model?.provider !== PROVIDER_ID) {
			ctx.ui.setStatus(PROVIDER_ID, undefined);
			return;
		}
		await refresh(ctx, true);
		const problem = warning();
		if (problem) ctx.ui.notify(problem.message, problem.level);
	});

	pi.on("session_start", async (_event, ctx) => {
		showStatus(ctx);

		// One message per session. A stack of warnings for a single root cause
		// trains users to ignore all of them.
		const problem = warning();

		// Stay quiet for users of other providers. The exception is a provider
		// with no models: nothing can select it, so silence would hide the cause
		// forever.
		if (!usesTinfoil(ctx) && discovery.models.length > 0) return;

		if (problem) {
			ctx.ui.notify(problem.message, problem.level);
			return;
		}
		if (autostartNote) ctx.ui.notify(`Tinfoil: ${autostartNote}`, "info");
		if (discovery.error) {
			ctx.ui.notify(`Tinfoil: model discovery failed (${discovery.error}).`, "warning");
		} else if (!discovery.models.length) {
			ctx.ui.notify("Tinfoil: the proxy served no usable chat models.", "warning");
		}
	});
}
