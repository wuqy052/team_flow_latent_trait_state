% take the up triangle of the original matrix as an array
function uptri = uptriangle(edist)
uptri = [];
for i = 2:size(edist,1)
    uptri = [uptri; edist(i+1:end,i)];
end

