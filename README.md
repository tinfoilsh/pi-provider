# Tinfoil provider for pi

Use [Tinfoil](https://tinfoil.sh)'s verifiably-private open models from the
[pi](https://pi.dev) coding agent. Inference runs inside hardware secure
enclaves that even Tinfoil cannot read into.

## Setup

1. Install the extension:

   ```bash
   pi install npm:@tinfoilsh/pi-provider
   ```

2. Start pi and set your API key:

   ```
   /login tinfoil
   ```

3. Pick a Tinfoil model with `/model`.

## How verification works

When pi starts, the extension uses the [`tinfoil`
SDK](https://github.com/tinfoilsh/tinfoil-js) to verify the inference
enclave: it checks the enclave's attestation, confirms the running code
against the release digest signed in Sigstore, and binds the attested key
to the live connection. Every request body is then encrypted end-to-end
with HPKE, so only the verified enclave can read it.

While a Tinfoil model is active, the footer shows: `Tinfoil verified` or
`Tinfoil unverified`. Run `/tinfoil` at any time to re-run the verification
from scratch and print the verification document. If verification fails,
the extension fails closed: requests are blocked until `/tinfoil` succeeds.
A failed verification cannot send your prompts anywhere.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `TINFOIL_API_KEY` | _(none)_ | Your `tk_…` key, for headless workflows. Does not need to be set if using `/login tinfoil`, the preferred login for everyday operation. |
