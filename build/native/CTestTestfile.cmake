# CMake generated Testfile for 
# Source directory: C:/dev/Powershell/amp/src/native
# Build directory: C:/dev/Powershell/amp/build/native
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(kpn_unit_test "C:/dev/Powershell/amp/build/native/Debug/kpn_unit_test.exe")
  set_tests_properties(kpn_unit_test PROPERTIES  _BACKTRACE_TRIPLES "C:/dev/Powershell/amp/src/native/CMakeLists.txt;66;add_test;C:/dev/Powershell/amp/src/native/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(kpn_unit_test "C:/dev/Powershell/amp/build/native/Release/kpn_unit_test.exe")
  set_tests_properties(kpn_unit_test PROPERTIES  _BACKTRACE_TRIPLES "C:/dev/Powershell/amp/src/native/CMakeLists.txt;66;add_test;C:/dev/Powershell/amp/src/native/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(kpn_unit_test "C:/dev/Powershell/amp/build/native/MinSizeRel/kpn_unit_test.exe")
  set_tests_properties(kpn_unit_test PROPERTIES  _BACKTRACE_TRIPLES "C:/dev/Powershell/amp/src/native/CMakeLists.txt;66;add_test;C:/dev/Powershell/amp/src/native/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(kpn_unit_test "C:/dev/Powershell/amp/build/native/RelWithDebInfo/kpn_unit_test.exe")
  set_tests_properties(kpn_unit_test PROPERTIES  _BACKTRACE_TRIPLES "C:/dev/Powershell/amp/src/native/CMakeLists.txt;66;add_test;C:/dev/Powershell/amp/src/native/CMakeLists.txt;0;")
else()
  add_test(kpn_unit_test NOT_AVAILABLE)
endif()
