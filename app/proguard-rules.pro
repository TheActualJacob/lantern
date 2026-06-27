# Keep ARCore classes (they are referenced reflectively / via JNI).
-keep class com.google.ar.core.** { *; }
-dontwarn com.google.ar.core.**
