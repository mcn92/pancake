# Authentication

Authentication in this product covers how clients prove who they are, how
credentials are issued and stored, and which requests require which scopes
before the server will accept them at all.

## API keys

API keys are issued per project from the dashboard and carry the project's
default scopes. Treat a key like a password: it is sent on every request in
the Authorization header and grants whatever its scopes allow.

```bash
# this comment must not become a heading or split the section
echo "rotate:" && date
```

### Rotating keys

Rotate a key by issuing a second key first, deploying it everywhere the old
key is used, and only then revoking the old key, so that no window exists in
which requests fail with an authorization error.

### Rotating keys

This duplicate heading exercises slug deduplication: frameworks suffix the
second occurrence, and the generated anchor must match that convention or
every deep link into this section lands on the wrong one.

## Webhooks {#custom-hooks}

Webhook payloads are signed with the project's signing secret, and the
signature arrives in a header the receiver must verify before trusting the
payload body or acting on the event it describes at any point.

## Reading `config.json` files

Configuration files are read once at boot and merged over the defaults, so
a key present in both places takes the file's value while unknown keys are
rejected with an error naming the offending path in the file.

## See [the upgrade guide](https://example.com/upgrade)

Link syntax in a heading must slug to the link's text, not its URL, since
that is what the rendered page shows and what the framework's own anchor
generator receives when it builds the id attribute for the heading.

## ACTION_QUERY_PARAMS

Snake case identifiers appear constantly in developer documentation and
their underscores are part of the identifier: stripping them would break
both the display text and the anchor that deep links depend on here.
