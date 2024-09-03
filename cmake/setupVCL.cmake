cpmaddpackage(
  NAME
  vcl
  GIT_REPOSITORY
  https://github.com/vectorclass/version2.git
  GIT_TAG
  v2.02.01
  DOWNLOAD_ONLY
  YES)

if(vcl_ADDED)
  add_library(vcl INTERFACE)
  target_include_directories(vcl INTERFACE ${vcl_SOURCE_DIR})
endif()
