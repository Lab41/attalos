require 'torch'

-- Stolen from: http://lua-users.org/wiki/SplitJoin
function string:split(sSeparator, nMax, bRegexp)
    assert(sSeparator ~= '')
    assert(nMax == nil or nMax >= 1)

    local aRecord = {}

    if self:len() > 0 then
        local bPlain = not bRegexp
        nMax = nMax or -1

        local nField, nStart = 1, 1
        local nFirst,nLast = self:find(sSeparator, nStart, bPlain)
        while nFirst and nMax ~= 0 do
            aRecord[nField] = self:sub(nStart, nFirst-1)
            nField = nField+1
            nStart = nLast+1
            nFirst,nLast = self:find(sSeparator, nStart, bPlain)
            nMax = nMax-1
        end
        aRecord[nField] = self:sub(nStart)
    end

    return aRecord
end

--[[
Read in a CSV file of words and vectors
--]]

function read(file)
    -- Open file
    local f = assert(io.open(file, "rb"))
    local word2vec = {}
    -- Read lines and assign to a tensor
    while true do
        local line = f:read("*line")
        if line == nil then break end

        local sline = string.split(line, ",")
        local key = sline[1]
        local len = table.getn(sline)
        local vec = {unpack(sline, 2, len)}
        local tensor = torch.Tensor(vec)

        word2vec[key] = tensor
    end
    f:close()

    return tensor
end

read("./test.txt")
