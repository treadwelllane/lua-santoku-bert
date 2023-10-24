package = "santoku-bert"
version = "0.0.1-1"
rockspec_format = "3.0"

source = {
  url = "git+ssh://git@github.com:treadwelllane/lua-santoku-bert.git",
  tag = version
}

description = {
  homepage = "https://github.com/treadwelllane/lua-santoku-bert",
  license = "MIT"
}

dependencies = {
  "lua >= 5.1",
}

test_dependencies = {
  "santoku >= 0.0.87-1",
  "luafilesystem >= 1.8.0-1",
  "luassert >= 1.9.0-1",
  "luacov >= 0.15.0",
}

build = {
  type = "make",
  makefile = "luarocks.mk",
  variables = {
    CC = "$(CC)",
    CFLAGS = "$(CFLAGS)",
    LIBFLAG = "$(LIBFLAG)",
    LIB_EXTENSION = "$(LIB_EXTENSION)",
  },
  install_variables = {
    INST_LIBDIR = "$(LIBDIR)",
    INST_LUADIR = "$(LUADIR)"
  }
}

test = {
  type = "command",
  command = "make -f luarocks.mk test",
}