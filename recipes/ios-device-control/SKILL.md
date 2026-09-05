---
name: ios-device-control
description: Control physical iPhones and iPads with pymobiledevice3. Use for USB device discovery, app launching, screenshots, touch gestures, and live screen viewing on a paired device. Simulator tasks use the iOS simulator tools instead.
---

# Physical iOS device control

Use the fleet-managed `pymobiledevice3` CLI on the host connected to the device.
The initial package supports Apple Silicon macOS with Python 3.13. Other hosts
can reach that Mac through SSH; Tailscale connectivity alone does not pair an iPhone.
Consult the fleet skill before changing user-level tools. The CLI belongs in
machine overlays; device pairing records stay local to the device host.

## Discover and connect

1. Run `pymobiledevice3 usbmux list` and select the device by name, product type,
   iOS version, and identifier. Do not assume the first device is the requested one.
2. Check the command's `--help` for connection selection. Lockdown commands and
   developer commands have different options; not every command accepts `--udid`.
3. On macOS, check the existing Apple pairing with `xcrun devicectl list devices`
   and `xcrun devicectl device info details --device <identifier>` when available.
   An already-paired native connection can work even when a separate lockdown
   client asks to pair. For an unpaired device, the user must unlock it and accept
   Trust. Stop dependent actions while that confirmation is pending.
4. Developer services need Developer Mode and a mounted Developer Disk Image.
   Check status before changing either. `pymobiledevice3 mounter auto-mount` mounts
   the matching image. Enabling Developer Mode can require a reboot and a device-side
   confirmation; explain the concrete change before proceeding.

On modern iOS, developer services use an RSD tunnel. On macOS, `--native` can
reuse Apple's existing tunnel without root. `--userspace` is another no-root path.
Set `PYMOBILEDEVICE3_UDID` to the discovered identifier when using `--native`.
Use explicit `--rsd HOST PORT` or a selected `--tunnel` when appropriate. Do not
silently fall back to another device after a connection failure.

A mounted DDI does not guarantee HID support. The Xcode 26.5 image (`17F42`)
exposed DVT screenshots and app launching but lacked the CoreDevice HID services
on the tested iPhone. Inspect advertised services before retrying gestures. Keep
the existing Xcode image available when testing a newer image, and restore it if
the replacement cannot mount. A DDI change is distinct from an iOS firmware update.

## Observe, act, verify

On iPhone 12 Pro / iOS 26.2.1, the tested `27A5228h` DDI supports screenshots,
app launching, and the Home hardware button. Direct touch starts a media stream,
which the device rejects with CoreDevice error 9021: "Remote control requires
iOS 27.0 or later on this device." Do not promise direct taps on iOS 26 from
the generic upstream command examples. A signed WebDriverAgent is a separate
fallback; it needs an appropriate existing signing setup and device installation.
An OS upgrade is a separate user decision, not a smoke-test repair.

## Signed runner for iOS 26

Use the official Appium WebDriverAgent source and an existing development team.
Read the team's identifier from the user's working Xcode project; never embed
personal team IDs or device identifiers in this shared skill. Give the runner
a unique bundle identifier, select `WebDriverAgentRunner` and the exact physical
device in Xcode, then use Product > Test. Keep that test running during control.

Do not conclude that signing keys are missing from `security find-identity` alone.
On a migrated Mac home, the command-line build reported no usable private key
while Keychain Access showed a valid certificate and private key. Building the
same runner in the Xcode GUI signed it with the existing identity. Verify the
result with `codesign -dv --verbose=2 <runner.app>`. Do not revoke a working
certificate or create another identity to work around a session-access failure.

The first XCTest launch may ask for the iPhone passcode to enable UI Automation.
Ask the user to enter it on the phone; do not request it in chat or infer that
an unavailable WDA port means the runner failed before inspecting the screen.

With the runner active and `PYMOBILEDEVICE3_UDID` set, use:

```sh
pymobiledevice3 developer wda status --native
pymobiledevice3 developer wda launch --native com.apple.calculator
pymobiledevice3 developer wda list-items --native --help
pymobiledevice3 developer wda tap --native --session-id <session> <selector>
pymobiledevice3 developer wda screenshot --native --help
```

Use selectors from `list-items`; do not guess localized button names. WDA talks
to the device directly, so these CLI commands do not require a local port forward.
Check help for `swipe`, `type`, and `press`. WDA coordinates use screen points,
unlike the normalized UInt16 coordinates of the CoreDevice API. An installed
runner can also be started with `--xctrunner <bundle-id>`, but the CLI stops that
runner after each command; keep Xcode's test active for a multi-step session.

Capture a fresh image, inspect it, choose one action, then capture and inspect
the result before selecting another action. A successful exit code is not proof:
pymobiledevice3 can log an error and exit zero. Verify the output file exists
and that its pixels show the requested state.

Examples (check the installed version's help when an option differs):

```sh
pymobiledevice3 developer core-device screen-capture screenshot --native screen.png
pymobiledevice3 developer dvt screenshot --native screen.png
pymobiledevice3 developer dvt launch --native com.apple.calculator
pymobiledevice3 developer core-device hid button --native home press
pymobiledevice3 developer core-device get-display-info --native
pymobiledevice3 developer core-device universal-hid-service tap --native -- 32768 32768
pymobiledevice3 developer core-device universal-hid-service drag --native -- 32768 5000 32768 60000
```

Touch coordinates are normalized UInt16 values, 0–65535 across the display.
Convert from inspected pixels using the current image dimensions and orientation.
Use `drag` for contact gestures: the command called `swipe` is pointer motion
without contact and may not scroll. Touch uses an automatically managed media stream.

For a smoke test, prefer Calculator and Home: capture the starting state, launch
Calculator, tap a simple known calculation, inspect its result, and return Home.
Save distinct numbered screenshots for each state. Keep images outside git because
they may show private content. Report which actions passed, which failed, and the
exact device/iOS/tool versions tested. Do not claim support for untested iOS versions.

## Live viewing

`developer core-device display serve-web` streams to a browser.
`developer core-device display serve-vnc` uses macOS VideoToolbox and works with
Screen Sharing. Keep listeners on loopback; remote access can use an SSH forward.
Check current CLI bind options before starting a listener. Stop test servers when done.

References:
- https://doronz88.github.io/pymobiledevice3/guides/cli-recipes/
- https://doronz88.github.io/pymobiledevice3/guides/ios17-tunnels/

## Apple signing on macOS

The fleet recipe uses verified HTTPS for Apple TSS. If Python cannot validate
Apple's chain while system curl can, use the existing macOS trust roots rather
than disabling verification. Export the public roots to a machine-local cache:

```sh
mkdir -p "$HOME/.cache/ios-device-control"
security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain \
  > "$HOME/.cache/ios-device-control/system-roots.pem"
REQUESTS_CA_BUNDLE="$HOME/.cache/ios-device-control/system-roots.pem" \
  pymobiledevice3 mounter auto-mount --udid <identifier>
```

An already-mounted image is not replaced by `auto-mount`. Inspect it first;
unmount deliberately when changing images. `mounter auto-mount --xcode
/Applications/Xcode.app` can restore the installed Xcode image.
