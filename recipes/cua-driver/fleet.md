
## Fleet-managed installation

On this fleet, `cua-driver` comes from ai-demos and this skill comes from
my-skill-forge. Update both release pins together in agent-fleet, publish the
packages, and run fleet sync. This replaces upstream installer, self-update,
and `skills install/update` instructions above; do not create a second install.
The fleet wrapper disables telemetry and independent update checks. On macOS,
fleet sync deploys the signed app to /Applications; OS grants still belong to
the user. On Linux, use the target user's graphical session and accessibility
bus. An SSH shell without that session is not desktop-ready.

For delegated desktop work, follow the delegation skill's Astra/low routing.
Allow one active computer-use worker per desktop session. Verify actions from
fresh application state; do not treat a successful command as visual proof.
