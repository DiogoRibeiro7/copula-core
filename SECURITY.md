# Security policy

## Supported versions

`copula-core` is pre-1.0. Only the latest published minor release receives fixes;
older releases are not patched.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability.

Report it privately through GitHub:
<https://github.com/DiogoRibeiro7/copula-core/security/advisories/new>

Include:

- a description of the issue and its impact
- affected versions or commit SHAs
- reproduction steps or a proof of concept

You can expect an acknowledgement once the report has been reviewed, questions if
the reproduction is incomplete, and coordination on disclosure timing for valid
reports. Fixes are released as a new version and, where appropriate, published as
a GitHub security advisory and a RustSec advisory.

## Scope

This is a numerical library with no network or file-system access and no `unsafe`
code. Reports that are likely in scope:

- panics, unbounded loops, or unbounded memory use reachable from public APIs
  with ordinary inputs (denial of service)
- silently wrong numerical results that could plausibly lead to materially wrong
  decisions

Vulnerabilities in dependencies should be reported upstream. If one affects this
crate in a specific way, a report here is welcome too.
