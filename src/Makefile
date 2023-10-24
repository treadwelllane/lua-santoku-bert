SRC_LUA = $(shell find * -name '*.lua')
SRC_C = $(shell find * -name '*.c')
SRC_O = $(SRC_C:.c=.o)
SRC_SO = $(SRC_O:.o=.$(LIB_EXTENSION))

INST_LUA = $(addprefix $(INST_LUADIR)/, $(SRC_LUA))
INST_SO = $(addprefix $(INST_LIBDIR)/, $(SRC_SO))

LIB_CFLAGS = -I../deps/python/cpython -I../deps/python/cpython/Include
LIB_LDFLAGS = -L../deps/python/cpython/ -lpython3.11

all: $(SRC_O) $(SRC_SO)

%.o: %.c
	$(CC) $(CFLAGS) $(LIB_CFLAGS) -c -o $@ $^

%.$(LIB_EXTENSION): %.o
	$(CC) $(LDFLAGS) $(LIB_LDFLAGS) $(LIBFLAG) -o $@ $^

install: $(INST_LUA) $(INST_SO)

$(INST_LUADIR)/%.lua: ./%.lua
	mkdir -p $(dir $@)
	cp $^ $@

$(INST_LIBDIR)/%.$(LIB_EXTENSION): ./%.$(LIB_EXTENSION)
	mkdir -p $(dir $@)
	cp $^ $@

.PHONY: all install
