# Theriad provider billing policy

Status: BYOK-only. Theriad does not currently offer subscriptions, checkout,
included model allowances, or a platform usage fee.

## OpenRouter

Users add their own OpenRouter API key in Account. OpenRouter bills the account
associated with that key at OpenRouter's current model rates, including any
provider limits or credits. Free models remain subject to OpenRouter's
availability and limits.

Theriad provides routing, projects, and cost visibility. It does not act as a
reseller of OpenRouter usage in the current live implementation.

## Future billing

The previous Stripe subscription implementation remains isolated in the codebase
for possible future work. Stripe environment variables and checkout are not part
of the current deployment contract.
