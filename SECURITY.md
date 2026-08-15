# Security Policy

## Supported versions

qorch is pre-1.0. Security fixes land on `main` and in the next release; there are no
maintained back-branches yet.

| Version | Supported |
|---|---|
| 0.1.x | ✅ |
| < 0.1 | ❌ |

## Reporting a vulnerability

**Do not open a public issue.**

Use GitHub's private vulnerability reporting — the **Report a vulnerability** button
under the repository's [Security tab](https://github.com/jjinfotechsolutionspvtltd-oss/qorch/security).
That opens a private channel visible only to maintainers.

If that is unavailable to you, email jjinfotechsolutionspvtltd@gmail.com with
`SECURITY` in the subject.

Please include: what the issue is, how to reproduce it (a minimal circuit or input
file is ideal), and what an attacker could achieve. We will acknowledge within 5
working days and keep you updated as we investigate.

## Scope

The core has no runtime dependencies and executes no network calls, which removes
most of the usual attack surface. What remains genuinely matters:

**In scope**

- **Deserialization.** `from_qmi` (binary), `from_json`, and `from_qasm3` parse
  externally-supplied bytes and text. Malformed input should raise a clean, typed
  error — never crash the interpreter, exhaust memory, or execute anything. This is
  the primary trust boundary.
- **CLI file handling.** Path traversal or unexpected behavior when reading
  circuit files.
- **Credential handling in optional backends.** The Qiskit/IBM adapter reads a token
  from the environment; a token leaking into logs, error messages, or serialized
  output is a valid report.
- **Dependency issues** in the optional extras that qorch's usage makes exploitable.

**Out of scope**

- Resource exhaustion from a circuit you deliberately built to be enormous.
  Simulating 40 qubits on a statevector backend is expected to be infeasible — that
  is physics, not a vulnerability.
- Numerical inaccuracy or an incorrect physical result. Those are important bugs, but
  they are correctness issues: please open a normal public issue with the circuit
  that reproduces them.
- Vulnerabilities in Qiskit, numpy, or other optional dependencies that do not
  involve how qorch calls them. Report those upstream.

## Disclosure

We aim to have a fix or a clear mitigation within 90 days of a confirmed report, and
we will credit reporters who want it. If you plan to publish, please coordinate the
timing with us.
