[app]
title = Thin Lens Simulator
package.name = thinlens
package.domain = com.example

# App version (required)
version = 0.1.1

source.dir = .
source.include_exts = py,kv,png,jpg,jpeg,bmp,gif,ttf,otf,txt

# If you keep assets in subfolders, uncomment:
# include_patterns = images/*, assets/*

requirements = python3,kivy==2.2.0,pillow,numpy,plyer
orientation = landscape
fullscreen = 0
log_level = 2

# Android 13+ target
android.api = 33
android.minapi = 29
android.ndk_api = 21

# Force Buildozer/p4a to use the preinstalled SDK/NDK and accept licenses
android.accept_sdk_license = True
android.sdk_path = /usr/local/lib/android/sdk
android.ndk_path = /usr/local/lib/android/sdk/ndk/25.2.9519653
android.ndk = 25.2.9519653
# Prefer stable build-tools to avoid preview prompts
android.build_tools_version = 34.0.0

# Media/image permissions (Android 13+ and legacy)
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_IMAGES

# Optional: ABIs (default builds both)
# android.archs = arm64-v8a, armeabi-v7a

# Optional icons/presplash
# icon.filename = %(source.dir)s/icon.png
# presplash.filename = %(source.dir)s/presplash.png

[buildozer]
log_level = 2
warn_on_root = 1