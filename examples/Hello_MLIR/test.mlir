module {
  %c1 = "builtin.unrealized_conversion_cast"() : () -> i32
  %c2 = "builtin.unrealized_conversion_cast"() : () -> i32
  %result = my.add(%c1, %c2) block 32 : (i32, i32) -> i32
}