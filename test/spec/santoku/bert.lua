local test = require("santoku.test")
local bert = require("santoku.bert")
local inspect = require("santoku.inspect")
local vec = require("santoku.vector")

test("bert", function ()

  local index = bert.index()
  local encoder = bert.encoder()

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

  corpus:each(function (document)
    local document_embedding = encoder:encode(document)
    index:add(document_embedding)
  end)

  local query_embedding = encoder:encode(query)

  local results = index:search(query_embedding, 3)

  print()
  print("Query: " .. query)
  print()

  for _, result in ipairs(results) do
    print(inspect(result))
  end

  print()

end)
