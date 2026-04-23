# vendored JS deps

`livekit-client.umd.min.js` is expected here (bundled locally so the
page doesn't depend on unpkg at runtime).

Fetch once on Atlas during deploy:

```bash
curl -o /home/claude/atlas-web/table/vendor/livekit-client.umd.min.js \
  https://unpkg.com/livekit-client@2/dist/livekit-client.umd.min.js
```

Committing the minified UMD itself to the repo is a judgment call —
in this repo we don't commit vendored minified libs, so the file is
downloaded at deploy time by the deploy script instead. See the
Phase A section of `docs/ARTEMIS_AVATAR_ROADMAP.md`.
