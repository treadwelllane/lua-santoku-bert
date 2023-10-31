local _ENV = {}

name = "santoku-bert"
version = "0.0.1-1"
variable_prefix = "TK_BERT"
license = "MIT"
public = true

dependencies = {
  "lua >= 5.1",
  "santoku >= 0.0.100-1",
  "santoku-python >= 0.0.9-1",
}

test_dependencies = {
  "luafilesystem >= 1.8.0-1",
  "luassert >= 1.9.0-1",
  "luacov >= 0.15.0-1",
  "luasocket >= 3.1.0-1",
  "inspect >= 3.1.3-0",
}

homepage = "https://github.com/treadwelllane/lua-" .. name
tarball = name .. "-" .. version .. ".tar.gz"
download = homepage .. "/releases/download/" .. version .. "/" .. tarball

return { env = _ENV }
