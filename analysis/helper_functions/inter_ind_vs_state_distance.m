% calculate inter individual (within state) / intra individual (between state) distance
% to quantify the conceptual diagram
function  [bwd] =inter_ind_vs_state_distance(subcondcentroid)

% 1. Inter-individual (within state) distance
% first filter out each state, then construct pairwise distance across
% individuals, finally average over 3 states
statelist = unique(subcondcentroid(:,2));
dist_inter = [];
for s = 1:3
    state = statelist{s};
    subLD_filtered = subcondcentroid(strcmp(subcondcentroid(:,2),state),:);
    dist_inter_state = squareform(pdist(cell2mat(subLD_filtered(:,3:5)),'euclidean'));
    for i = 1:size(dist_inter_state,1)-1
        dist_inter = [dist_inter; dist_inter_state(i+1:end,i)];
    end
end

% 2. Inter-state (intra-individual) distance
% first filter out each subject, then pairwise distance across states,
% finally average over subjects
sublist = unique(subcondcentroid(:,1));
dist_intra = [];
for p = 1:20
    sub = sublist{p};
    stateLD_filtered = subcondcentroid(strcmp(subcondcentroid(:,1),sub),:);
    dist_intra_sub = squareform(pdist(cell2mat(stateLD_filtered(:,3:5)),'euclidean'));
    for i = 1:size(dist_intra_sub,1)-1
        dist_intra = [dist_intra; dist_intra_sub(i+1:end,i)];
    end
end

bwd = mean(dist_inter) / mean(dist_intra);

