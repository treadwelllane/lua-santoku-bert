local py = require("santoku.python")
local compat = require("santoku.compat")
local vec = require("santoku.vector")

py.open("libpython3.11.so")

local faiss = py.import("faiss")
local sbert = py.import("sentence_transformers")

local M = {}

M.DEFAULT_DIMENSIONS = 384
M.DEFAULT_MODEL = "all-MiniLM-L6-v2"
M.DEFAULT_INDEX = "IndexFlatL2"

M.IDX_ENCODER = {}
M.IDX_INDEX = {}

M.MT_ENCODER = { __index = M.IDX_ENCODER, __name = "santoku_bert_encoder" }
M.MT_INDEX = { __index = M.IDX_INDEX, __name = "santoku_bert_index" }
M.MT_EMBEDDING = { _name = "santoku_bert_embedding" }

M.index = function (name, dimensions)
  local idx = {}
  idx.name = name or M.DEFAULT_INDEX
  idx.n = 0
  idx.dimensions = dimensions or M.DEFAULT_DIMENSIONS
  idx.index = faiss[idx.name](idx.dimensions)
  return setmetatable(idx, M.MT_INDEX)
end

M.encoder = function (model, dimensions)
  local enc = {}
  enc.model = model or M.DEFAULT_MODEL
  enc.dimensions = dimensions or M.DEFAULT_DIMENSIONS
  enc.encoder = sbert.SentenceTransformer(enc.model)
  return setmetatable(enc, M.MT_ENCODER)
end

M.IDX_ENCODER.encode = function (enc, str)
  assert(compat.hasmeta(enc, M.MT_ENCODER))
  assert(compat.istype.string(str))
  local embd = enc.encoder.encode(str, py.kwargs({
    convert_to_numpy = true,
    normalize_embeddings = true
  })).reshape(1, enc.dimensions)
  return setmetatable({ embd = embd }, M.MT_EMBEDDING)
end

M.IDX_INDEX.add = function (idx, embd)
  assert(compat.hasmeta(idx, M.MT_INDEX))
  assert(compat.hasmeta(embd, M.MT_EMBEDDING))
  idx.index.add(embd.embd)
  idx.n = idx.n + 1
  return idx.n
end

M.IDX_INDEX.search = function (idx, query, n)
  assert(compat.hasmeta(idx, M.MT_INDEX))
  assert(compat.hasmeta(query, M.MT_EMBEDDING))
  assert(compat.istype.number(n))
  assert(compat.gt(0, n))
  local res = idx.index.search(query.embd, n)
  local ranks, ids = res[0], res[1]
  local ret = vec()
  for i = 0, ids.shape[1] - 1 do
    ret:append({
      document = ids.item(0, i),
      distance = ranks.item(0, i)
    })
  end
  return ret
end

return M
