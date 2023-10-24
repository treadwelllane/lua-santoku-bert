// TODO: Py_DECREF where needed

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "lua.h"
#include "lauxlib.h"
#include "dlfcn.h"

#define TK_BERT_DIMENSIONS 384
#define TK_BERT_MODEL "all-MiniLM-L6-v2"

#define TK_BERT_MTIDX "santoku_bert_index"
#define TK_BERT_MTENC "santoku_bert_encoder"
#define TK_BERT_MTEMB "santoku_bert_embedding"

int tk_bert_pyerr (lua_State *L)
{
  PyObject *ptype, *pvalue, *ptrace;
  PyErr_Fetch(&ptype, &pvalue, &ptrace);
  luaL_error(L, PyUnicode_AsUTF8(PyObject_Str(pvalue)));
  return 0;
}

int tk_bert_mtenc_destroy (lua_State *L)
{
  return 0;
}

int tk_bert_mtenc_encode (lua_State *L)
{
  luaL_checkudata(L, -2, TK_BERT_MTENC);
  luaL_checkstring(L, -1);

  lua_getiuservalue(L, -2, 1);
  PyObject *mod = lua_touserdata(L, -1);
  lua_pop(L, 1);

  size_t len;
  const char *s = lua_tolstring(L, -1, &len);

  PyObject *enc = PyObject_GetAttrString(mod, "encode");
  if (!enc) return tk_bert_pyerr(L);

  PyObject *args = Py_BuildValue("(s#)", s, len);
  if (!args) return tk_bert_pyerr(L);

  PyObject *kwargs = PyDict_New();
  if (!kwargs) return tk_bert_pyerr(L);

  if (PyDict_SetItemString(kwargs, "convert_to_numpy", Py_True))
    return tk_bert_pyerr(L);

  if (PyDict_SetItemString(kwargs, "normalize_embeddings", Py_True))
    return tk_bert_pyerr(L);

  PyObject *embd = PyObject_Call(enc, args, kwargs);
  if (!embd) return tk_bert_pyerr(L);

  embd = PyObject_CallMethod(embd, "reshape", "i i", 1, TK_BERT_DIMENSIONS);
  if (!embd) return tk_bert_pyerr(L);

  lua_newuserdatauv(L, 0, 1);
  lua_pushlightuserdata(L, embd);
  lua_setiuservalue(L, -2, 1);
  luaL_setmetatable(L, TK_BERT_MTEMB);

  return 1;
}

int tk_bert_mt_encoder (lua_State *L)
{
  PyObject *lib = PyImport_ImportModule("sentence_transformers");
  if (!lib) return tk_bert_pyerr(L);

  PyObject *st = PyObject_GetAttrString(lib, "SentenceTransformer");
  if (!st) return tk_bert_pyerr(L);

  PyObject *str = Py_BuildValue("s", TK_BERT_MODEL);
  if (!str) return tk_bert_pyerr(L);

  PyObject *model = PyObject_CallOneArg(st, str);
  if (!model) return tk_bert_pyerr(L);

  lua_newuserdatauv(L, 0, 1);
  lua_pushlightuserdata(L, model);
  lua_setiuservalue(L, -2, 1);
  luaL_setmetatable(L, TK_BERT_MTENC);

  return 1;
}

int tk_bert_mtidx_destroy (lua_State *L)
{
  return 0;
}

int tk_bert_mtidx_add (lua_State *L)
{
  luaL_checkudata(L, -2, TK_BERT_MTIDX);
  luaL_checkudata(L, -1, TK_BERT_MTEMB);

  lua_getiuservalue(L, -2, 1);
  PyObject *idx = lua_touserdata(L, -1);
  lua_pop(L, 1);

  lua_getiuservalue(L, -1, 1);
  PyObject *embd = lua_touserdata(L, -1);
  lua_pop(L, 1);

  PyObject *res = PyObject_CallMethod(idx, "add", "O", embd);
  if (!res) return tk_bert_pyerr(L);

  return 0;
}

int tk_bert_mtidx_search (lua_State *L)
{
  luaL_checkudata(L, -3, TK_BERT_MTIDX);
  luaL_checkudata(L, -2, TK_BERT_MTEMB);
  luaL_checktype(L, -1, LUA_TNUMBER);

  lua_getiuservalue(L, -3, 1);
  PyObject *idx = lua_touserdata(L, -1);
  lua_pop(L, 1);

  lua_getiuservalue(L, -2, 1);
  PyObject *embd = lua_touserdata(L, -1);
  lua_pop(L, 1);

  int k = lua_tointeger(L, -1);

  PyObject *res = PyObject_CallMethod(idx, "search", "O i", embd, k);
  if (!res) return tk_bert_pyerr(L);

  PyObject *ranks = PyTuple_GetItem(res, 0);
  if (!ranks) return tk_bert_pyerr(L);

  PyObject *ids = PyTuple_GetItem(res, 1);
  if (!ranks) return tk_bert_pyerr(L);

  PyObject *shape = PyObject_GetAttrString(ranks, "shape");
  if (!shape) return tk_bert_pyerr(L);

  PyObject *olen = PyTuple_GetItem(shape, 1);
  if (!olen) return tk_bert_pyerr(L);

  PyObject *llen = PyNumber_Long(olen);
  if (!olen) return tk_bert_pyerr(L);

  long len = PyLong_AsLong(llen);

  lua_newtable(L);

  for (long i = 0; i < len; i ++) {

    lua_pushinteger(L, i + 1);

    lua_newtable(L);

    PyObject *okey = PyObject_CallMethod(ids, "item", "i i", 0, i);
    if (!okey) return tk_bert_pyerr(L);

    PyObject *fkey = PyNumber_Long(okey);
    if (!fkey) return tk_bert_pyerr(L);

    long key = PyLong_AsLong(fkey);
    lua_pushinteger(L, key + 1);
    lua_setfield(L, -2, "document");

    PyObject *oval = PyObject_CallMethod(ranks, "item", "i i", 0, i);
    if (!oval) return tk_bert_pyerr(L);

    PyObject *fval = PyNumber_Float(oval);
    if (!fval) return tk_bert_pyerr(L);

    double val = PyFloat_AsDouble(fval);
    lua_pushnumber(L, val);
    lua_setfield(L, -2, "distance");

    lua_settable(L, -3);

  }

  return 1;
}

int tk_bert_mt_index (lua_State *L)
{
  PyObject *lib = PyImport_ImportModule("faiss");
  if (!lib) return tk_bert_pyerr(L);

  PyObject *idx = PyObject_GetAttrString(lib, "IndexFlatL2");
  if (!idx) return tk_bert_pyerr(L);

  PyObject *d = Py_BuildValue("i", TK_BERT_DIMENSIONS);
  if (!d) return tk_bert_pyerr(L);

  PyObject *model = PyObject_CallOneArg(idx, d);
  if (!model) return tk_bert_pyerr(L);

  lua_newuserdatauv(L, 0, 1);
  lua_pushlightuserdata(L, model);
  lua_setiuservalue(L, -2, 1);
  luaL_setmetatable(L, TK_BERT_MTIDX);

  return 1;
}

int tk_bert_mtemb_destroy (lua_State *L)
{
  return 0;
}

luaL_Reg tk_bert_mtenc_fns[] =
{
  { "encode", tk_bert_mtenc_encode },
  { NULL, NULL }
};

luaL_Reg tk_bert_mtidx_fns[] =
{
  { "search", tk_bert_mtidx_search },
  { "add", tk_bert_mtidx_add },
  { NULL, NULL }
};

luaL_Reg tk_bert_mtemb_fns[] =
{
  { NULL, NULL }
};

luaL_Reg tk_bert_mt_fns[] =
{
  { "encoder", tk_bert_mt_encoder },
  { "index", tk_bert_mt_index },
  { NULL, NULL }
};

void tk_bert_setup_python (lua_State *L)
{
  void *python = dlopen("libpython3.11.so", RTLD_NOW | RTLD_GLOBAL);

  if (python == NULL)
    luaL_error(L, "Error loading python library");

  Py_Initialize();
}

int luaopen_santoku_bert_lib (lua_State *L)
{
  lua_newtable(L); // mt
  luaL_setfuncs(L, tk_bert_mt_fns, 0); // mt

  luaL_newmetatable(L, TK_BERT_MTENC); // mt mte
  lua_newtable(L); // mt mte idx
  luaL_setfuncs(L, tk_bert_mtenc_fns, 0); // mt mte idx
  lua_setfield(L, -2, "__index"); // mt mte
  lua_pushcfunction(L, tk_bert_mtenc_destroy); // mt mte fn
  lua_setfield(L, -2, "__gc"); // mt mte
  lua_pop(L, 1); // mt

  luaL_newmetatable(L, TK_BERT_MTIDX); // mt mte
  lua_newtable(L); // mt mte idx
  luaL_setfuncs(L, tk_bert_mtidx_fns, 0); // mt mte idx
  lua_setfield(L, -2, "__index"); // mt mte
  lua_pushcfunction(L, tk_bert_mtidx_destroy); // mt mte fn
  lua_setfield(L, -2, "__gc"); // mt mte
  lua_pop(L, 1); // mt

  luaL_newmetatable(L, TK_BERT_MTEMB); // mt mte
  lua_newtable(L); // mt mte idx
  luaL_setfuncs(L, tk_bert_mtemb_fns, 0); // mt mte idx
  lua_setfield(L, -2, "__index"); // mt mte
  lua_pushcfunction(L, tk_bert_mtemb_destroy); // mt mte fn
  lua_setfield(L, -2, "__gc"); // mt mte
  lua_pop(L, 1); // mt

  tk_bert_setup_python(L);

  return 1;
}
