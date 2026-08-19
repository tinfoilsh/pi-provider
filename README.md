# Tinfoil provider for pi

Use [Tinfoil](https://tinfoil.sh)'s verifiably-private open models from the
[pi](https://pi.dev) coding agent. Inference runs inside hardware secure
enclaves that even Tinfoil cannot read into.

## Setup

1. Install the Tinfoil Proxy. See
   [tinfoil.sh/coding-agents](https://tinfoil.sh/coding-agents), or build it
   from [source](https://github.com/tinfoilsh/tinfoil-proxy).

2. Install the extension:

   ```bash
   pi install npm:@tinfoilsh/pi-provider
   ```

3. Start pi and set your API key:

   ```
   /login tinfoil
   ```

4. Pick a Tinfoil model with `/model`.

If the proxy is not running, this extension starts it for you.

## Why the proxy

Tinfoil's privacy guarantee depends on a client that checks the enclave's
attestation as it connects. Pi cannot do that on its own. The proxy can, so
this extension always sends your requests through it.

Run `/tinfoil` at any time to see what the proxy verified. The footer shows a
short version while a Tinfoil model is active.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `TINFOIL_API_KEY` | _(none)_ | Your `tk_…` key. `/login tinfoil` is easier, because it survives a restart. |
| `TINFOIL_BASE_URL` | `http://127.0.0.1:3301/v1` | Change this only if the proxy uses a different local port. Addresses that are not local are refused. |
| `TINFOIL_AUTOSTART` | `1` | Set to `0` to never start the proxy. |
