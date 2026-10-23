# Headless Android emulator (API 34, Google APIs, x86_64) for instrumented tests.
# Built and run by start.sh; nothing here talks to anything but dl.google.com and Debian mirrors.
#
# Packages are pinned by exact archive + sha1 from Google's SDK repository manifests:
#   https://dl.google.com/android/repository/repository2-3.xml            (emulator, channel-0 = stable)
#   https://dl.google.com/android/repository/sys-img/google_apis/sys-img2-3.xml
# To bump: pick the new <complete><url>/<checksum> from those files and update the ARGs.
FROM debian:12-slim@sha256:7ff6ba31527b9e336162fa061746c366c4b2032f4d2cff2078bea43f78efe3f8

ARG EMULATOR_ZIP=emulator-linux_x64-16428233.zip
ARG EMULATOR_SHA1=cd7362ea55dfb86a418958138dc396e74165dd01
ARG SYSIMG_ZIP=x86_64-34_r14.zip
ARG SYSIMG_SHA1=e0f6c9a0691aa27bd597d0deb1bcfdc943ac8ca7
ARG API_LEVEL=34

# Runtime libraries the emulator binary links against even with -no-window, plus socat
# (the emulator only listens on loopback; socat re-exposes adb/console on the container IP).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl unzip socat \
      libdbus-1-3 libfontconfig1 libgl1 libpulse0 libnss3 libx11-6 libx11-xcb1 libxcb1 \
      libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxkbfile1 \
      libxrandr2 libxtst6 libtinfo6 zlib1g \
 && rm -rf /var/lib/apt/lists/*

ENV ANDROID_SDK_ROOT=/opt/android-sdk \
    ANDROID_HOME=/opt/android-sdk

# One RUN so the zips never land in a layer.
RUN set -eu; \
    mkdir -p "$ANDROID_SDK_ROOT/platforms" "$ANDROID_SDK_ROOT/platform-tools" \
             "$ANDROID_SDK_ROOT/system-images/android-$API_LEVEL/google_apis"; \
    cd /tmp; \
    curl -fsSLo emu.zip "https://dl.google.com/android/repository/$EMULATOR_ZIP"; \
    echo "$EMULATOR_SHA1  emu.zip" | sha1sum -c -; \
    unzip -q emu.zip -d "$ANDROID_SDK_ROOT"; \
    rm emu.zip; \
    curl -fsSLo sys.zip "https://dl.google.com/android/repository/sys-img/google_apis/$SYSIMG_ZIP"; \
    echo "$SYSIMG_SHA1  sys.zip" | sha1sum -c -; \
    unzip -q sys.zip -d "$ANDROID_SDK_ROOT/system-images/android-$API_LEVEL/google_apis"; \
    rm sys.zip; \
    test -f "$ANDROID_SDK_ROOT/system-images/android-$API_LEVEL/google_apis/x86_64/system.img"

COPY entrypoint.sh /usr/local/bin/gv-emulator-entrypoint
ENV API_LEVEL=${API_LEVEL}
EXPOSE 5554 5555
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/local/bin/gv-emulator-entrypoint"]
