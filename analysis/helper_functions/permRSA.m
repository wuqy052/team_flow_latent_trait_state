% permutation test for RSA
function [r_news, r0, p] = permRSA(mat1, mat2, n_perm)
% mat1 and mat2 are symmetrical matrices of the same size
% permutation: switch rows/ columns
r0 = corr(uptriangle(mat2), uptriangle(mat1));
for i = 1:n_perm
    sizemat = size(mat1, 1);
    order = randperm(sizemat );
    mat2_new = mat2(:, order);
    mat2_new = mat2_new(order,:);
    r_news(i) = corr(uptriangle(mat2_new), uptriangle(mat1));
end
p = sum(r0 < r_news) / n_perm;
% plot the figure
figure()
histogram(r_news, 100,'Edgecolor', 'none');
y1=get(gca,'ylim');
hold on
plot([r0, r0],y1);
title(['p = ', num2str(p)]);