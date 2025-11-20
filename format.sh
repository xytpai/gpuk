find ./ -type f \( -name "*.cpp" -o -name "*.h" \) -exec clang-format -i {} +
