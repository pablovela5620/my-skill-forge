---
name: ios-device-control
description: iOS device control for physical iPhones and iPads through pymobiledevice3 — use for device discovery, app launch, screenshots, touch gestures, or live screen viewing on a paired device. Simulators use the iOS simulator tools.
---

# Physical iOS device control

Run the fleet-managed `pymobiledevice3` CLI on the host the device is plugged
into. The fleet package supports Apple Silicon macOS with Python 3.13. Other
hosts reach that Mac through SSH; Tailscale connectivity alone does not pair an
iPhone. The CLI belongs in machine overlays (the fleet skill owns user-level
tool changes), and device pairing records stay local to the device host.

Each pymobiledevice3 command group takes its own connection options: check the
command's `--help` before use, and when an option below differs in the
installed version, follow `--help`.

## 1. Connect

1. Run `pymobiledevice3 usbmux list` and select the requested device by name,
   product type, iOS version, and identifier; the first device listed may be a
   different one. Not every command accepts `--udid`: lockdown and developer
   commands take different options.
2. On macOS, check the existing Apple pairing with `xcrun devicectl list devices`
   and `xcrun devicectl device info details --device <identifier>` when
   available. An already-paired native connection can work even when a separate
   lockdown client asks to pair. An unpaired device needs the user to unlock it
   and accept Trust; hold dependent actions until they confirm.
3. Developer services need Developer Mode and a mounted Developer Disk Image
   (DDI). Check both before changing either. `pymobiledevice3 mounter
   auto-mount` mounts the matching image (see DDI and Apple signing below).
   Enabling Developer Mode can require a reboot and a device-side confirmation:
   explain the concrete change before you proceed.
4. Choose the tunnel. Modern iOS runs developer services over an RSD tunnel. On
   macOS, `--native` reuses Apple's existing tunnel without root; set
   `PYMOBILEDEVICE3_UDID` to the discovered identifier when you use it.
   `--userspace` is another no-root path. Use an explicit `--rsd HOST PORT` or a
   selected `--tunnel` when appropriate.

If a connection fails, report it against the selected device; switching
silently to another device would act on the wrong phone.

**Done when** a read-only developer command (a screenshot) succeeds on the
selected device.

## 2. Observe, act, verify

Capture a fresh image, inspect it, choose one action, then capture and inspect
the result before choosing the next. A successful exit code is not proof:
pymobiledevice3 can log an error and exit zero. Verify that the output file
exists and that its pixels show the requested state.

```sh
pymobiledevice3 developer core-device screen-capture screenshot --native screen.png
pymobiledevice3 developer dvt screenshot --native screen.png
pymobiledevice3 developer dvt launch --native com.apple.calculator
pymobiledevice3 developer core-device hid button --native home press
pymobiledevice3 developer core-device get-display-info --native
```

Keep screenshots outside git: they may show private content. Save distinct,
numbered screenshots for each state.

## 3. Touch

Touch support depends on the iOS version and the DDI's services, so pick the
path from what the device advertises.

### CoreDevice HID

The device must accept CoreDevice remote control; the tested iPhone on iOS 26
rejected it (next section).

```sh
pymobiledevice3 developer core-device universal-hid-service tap --native -- 32768 32768
pymobiledevice3 developer core-device universal-hid-service drag --native -- 32768 5000 32768 60000
```

Coordinates are normalized UInt16 values, 0–65535 across the display. Convert
from inspected pixels using the current image dimensions and orientation. Use
`drag` for contact gestures: the command called `swipe` moves the pointer
without contact and may not scroll. Touch runs over an automatically managed
media stream.

A mounted DDI does not guarantee HID support. The Xcode 26.5 image (`17F42`)
exposed DVT screenshots and app launching but lacked the CoreDevice HID
services on the tested iPhone. Inspect advertised services before retrying
gestures.

### iOS 26: signed WebDriverAgent runner

On iPhone 12 Pro / iOS 26.2.1, the tested `27A5228h` DDI supports screenshots,
app launching, and the Home button. Direct touch starts a media stream that the
device rejects with CoreDevice error 9021: "Remote control requires iOS 27.0
or later on this device." The generic upstream touch examples therefore do not
work on iOS 26; touch there goes through a signed WebDriverAgent (WDA) runner,
which needs an existing signing setup and a device installation. An OS upgrade
is a separate user decision, not a smoke-test repair.

1. Use the official Appium WebDriverAgent source and an existing development
   team. Read the team identifier from the user's working Xcode project; keep
   personal team IDs and device identifiers out of this shared skill.
2. Give the runner a unique bundle identifier, select `WebDriverAgentRunner`
   and the exact physical device in Xcode, then use Product > Test. Keep that
   test running during control.
3. If the command-line build reports no usable private key, build the same
   runner in the Xcode GUI. On a migrated Mac home, `security find-identity`
   reported no usable key while Keychain Access showed a valid certificate and
   private key, and the GUI build signed with the existing identity. Verify with
   `codesign -dv --verbose=2 <runner.app>`. Keep the working certificate:
   revoking it or creating another identity does not fix a session-access
   failure.
4. The first XCTest launch may ask for the iPhone passcode to enable UI
   Automation. Ask the user to enter it on the phone; never request the
   passcode in chat. Inspect the screen before concluding that an unavailable
   WDA port means the runner failed.

With the runner active and `PYMOBILEDEVICE3_UDID` set:

```sh
pymobiledevice3 developer wda status --native
pymobiledevice3 developer wda launch --native com.apple.calculator
pymobiledevice3 developer wda list-items --native --help
pymobiledevice3 developer wda tap --native --session-id <session> <selector>
pymobiledevice3 developer wda screenshot --native --help
```

Take selectors from `list-items`; localized button names differ from what you
would guess. WDA talks to the device directly, so these commands need no local
port forward. Check `--help` for `swipe`, `type`, and `press`. WDA coordinates
are screen points, unlike the normalized UInt16 coordinates of CoreDevice. An
installed runner can also start with `--xctrunner <bundle-id>`, but the CLI
stops that runner after each command; keep Xcode's test active for a
multi-step session.

## 4. Smoke test and report

Use Calculator and Home: capture the starting state, launch Calculator, tap a
simple known calculation, inspect its result, and return Home, with a numbered
screenshot per state.

**Done when** the report names which actions passed, which failed, and the
exact device, iOS, DDI, and tool versions tested. Claim support only for the
iOS versions you tested.

## Live viewing

`developer core-device display serve-web` streams to a browser.
`developer core-device display serve-vnc` uses macOS VideoToolbox and works
with Screen Sharing. Bind listeners to loopback; remote access goes through an
SSH forward. Check the current CLI bind options before starting a listener, and
stop test servers when done.

## DDI and Apple signing

The fleet recipe uses verified HTTPS for Apple TSS. If Python cannot validate
Apple's chain while system curl can, point it at the existing macOS trust roots
and keep verification on. Export the public roots to a machine-local cache:

```sh
mkdir -p "$HOME/.cache/ios-device-control"
security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain \
  > "$HOME/.cache/ios-device-control/system-roots.pem"
REQUESTS_CA_BUNDLE="$HOME/.cache/ios-device-control/system-roots.pem" \
  pymobiledevice3 mounter auto-mount --udid <identifier>
```

`auto-mount` leaves an already-mounted image in place. Inspect it first, and
unmount deliberately when changing images. When testing a newer image, keep
the existing Xcode image available and restore it if the replacement cannot
mount: `mounter auto-mount --xcode /Applications/Xcode.app` restores the
installed Xcode image. A DDI change is distinct from an iOS firmware update.

References:
- https://doronz88.github.io/pymobiledevice3/guides/cli-recipes/
- https://doronz88.github.io/pymobiledevice3/guides/ios17-tunnels/
