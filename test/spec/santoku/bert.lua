local sock = require("socket")
local test = require("santoku.test")
local inspect = require("santoku.inspect")
local vec = require("santoku.vector")

test("bert", function ()

  local start, stop

  print()
  print("Importing...")

  start = sock.gettime()
  local bert = require("santoku.bert")
  stop = sock.gettime()
  local import_time = stop - start

  print("Loading...")

  start = sock.gettime()
  local index = bert.index()
  local encoder = bert.encoder()
  stop = sock.gettime()
  local load_time = stop - start

  local query = "Food-related, mentioning family members and travel"

  local corpus = vec.wrap({
    [[ Yeah, I think it's a good environment for learning English. ]],
    [[ The best key lime pie is still up for debate. ]],
    [[ I’m working on a sweet potato farm. ]],
    [[ I am my aunt's sister's daughter. ]],
    [[ She had that tint of craziness in her soul. ]],
    [[ Flesh-colored yoga pants were far worse than even he feared. ]],
    [[ She saw no irony asking me to change but wanting me to accept her for who she is. ]],
    [[ Henry couldn't decide if he was an auto mechanic or a priest. ]],
    [[ We have never been to Asia, nor have we visited Africa. ]],
    [[ I used to practice weaving with spaghetti. ]]
  })

  local encode_times = vec()

  print("Encoding...")

  corpus:each(function (document)
    start = sock.gettime()
    local document_embedding = encoder:encode(document)
    stop = sock.gettime()
    encode_times:append(stop - start)
    index:add(document_embedding)
  end)

  start = sock.gettime()
  local query_embedding = encoder:encode(query)
  stop = sock.gettime()
  encode_times:append(stop - start)

  start = sock.gettime()
  local results = index:search(query_embedding, 3)
  stop = sock.gettime()
  local search_time = stop - start

  print("Searching...")
  print()

  for _, result in ipairs(results) do
    print(inspect(result))
  end

  print()
  print("Import time: ", import_time)
  print("Load time: ", load_time)
  print("Encode time: ", encode_times:mean())
  print("Search time: ", search_time)
  print()

end)
