% funciton to extract dissimilarity matrix
function [edist_new] = extract_dissimilar(edist, subidx, xvalues)
edist_new = {};

for i = 1:size(xvalues,1)
    for j=1:size(xvalues,1)
        edist_new{i,j} = [];
    end
end
for p = 1:size(edist,1)
    for q = 1:size(edist,2)
        sub1 = subidx(p);
        sub2 = subidx(q);
        edist_new{sub1, sub2}(end+1) = edist(p,q);
    end
end
for i = 1:size(xvalues,1)
    for j=1:size(xvalues,1)
        edist_new{i,j} = mean(edist_new{i,j});
    end
end
edist_new = cell2mat(edist_new);

yvalues = xvalues;
corrmatp = [edist_new, zeros(size(edist_new,1),1)];
corrmatp = [corrmatp; zeros(1., size(corrmatp,2))];
pcolor(corrmatp);
ax = gca;
...
%// set labels
set(ax,'XTickLabel',xvalues)
set(ax,'YTickLabel',yvalues)
set(ax,'XTick', (1:size(edist_new,2))+0.5 )
xtickangle(45)
set(ax,'YTick', (1:size(edist_new,1))+0.5 )
