# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /Users/qqcao/Library/Android/sdk/tools/proguard/proguard-android.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

#kryo
-dontwarn sun.reflect.**
-dontwarn sun.misc.**
-dontwarn java.time.**
-dontwarn java.lang.invoke.**

-dontwarn java.beans.**
-keep,allowshrinking class com.esotericsoftware.** {
   <fields>;
   <methods>;
}
-keep,allowshrinking class java.beans.** { *; }
-keep,allowshrinking class sun.reflect.** { *; }
-keep,allowshrinking class com.esotericsoftware.kryo.** { *; }
-keep,allowshrinking class com.esotericsoftware.kryo.io.** { *; }
-keep class com.esotericsoftware.kryo.serializers.** { *; }
-dontwarn com.esotericsoftware.kryo.serializers.**
-keep,allowshrinking class sun.nio.ch.** { *; }
-dontwarn sun.nio.ch.**
