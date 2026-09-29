pluginManagement {
    repositories {
        google()
        maven("https://maven.aliyun.com/repository/google")        // mirror (Google's Android repo is blocked in some regions)
        maven("https://maven.aliyun.com/repository/gradle-plugin")
        gradlePluginPortal()
        mavenCentral()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        maven("https://maven.aliyun.com/repository/google")
        mavenCentral()
    }
}
rootProject.name = "PlateReader"
include(":core", ":app")
