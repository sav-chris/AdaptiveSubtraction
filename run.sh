#!/bin/bash

# Actiavte virtual environment
./venv/bin/activate 

# Update time stamps of build files
find /app/build -type f -exec touch {} +

# Run conan
conan install /app --build=missing 

# Run cmake
cd /app/build/Release/generators/ 
cmake /app -DCMAKE_TOOLCHAIN_FILE='/app/build/Release/generators/conan_toolchain.cmake' -DCMAKE_BUILD_TYPE='Release' 
cmake --build . --config Release 

# Run Tests
/app/build/Release/generators/adaptiveSubtraction.Test 

# Run main program
mv /app/build/Release/generators/AdaptiveSubtractionMain /app/release/AdaptiveSubtractionMain 
cd /app/release/ 
/app/release/AdaptiveSubtractionMain

