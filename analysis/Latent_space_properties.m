% Analyses after the latent space identification
% Figure 3 - Figure 6
addpath(genpath(pwd))
ld=load('../data/condsubldspace.mat');
ldspace = sortrows(ld.ldspace,2);
subidx = grp2idx(ldspace(:,2));
LD_table = cell2table(ld.ldspace(:,1:10),...
    "VariableNames",["condition" "participation" "trial" "LD1" "LD2" "LD3" "LD4" "LD5" "LD6" "LD7"]);

%% Figure 3D Test the trait-state hierarchy

% calculate centroid of each subject
subnames = unique(ld.ldspace(:,2));
condlist = {'tm','tmoc','tmrv'};
k=0;
subcondcentroid = {};
for i=1:size(subnames,1)
    ldspace_temp = ld.ldspace(strcmp(ld.ldspace(:,2), subnames{i}),:);
    for cond = 1:3
        k=k+1;
        ldspace_temp_cond = ldspace_temp(strcmp(ldspace_temp(:,1), condlist{cond}),:);
        subcondcentroid{k,1} = subnames{i};
        subcondcentroid{k,2} = condlist{cond};
        temp_centroid =  mean(cell2mat(ldspace_temp_cond(:,4:8)), 1);
        subcondcentroid{k,3} =temp_centroid(1);
        subcondcentroid{k,4} =temp_centroid(2);
        subcondcentroid{k,5} =temp_centroid(3);
    end
end

% calculate the actual between vs within subject distance
bwd0 =inter_ind_vs_state_distance(subcondcentroid);

% permutation
% shuffle the subject-condition label (perm(60))
% we hypothesis that the between subject distance should be larger than
% within subject distance (i.e., within each subject, across 3 task
% conditions), given random controls
n_perm = 5000;
for n=1:n_perm
    c = randperm(60);
    subcondcentroid_perm = subcondcentroid;
    subcondcentroid_perm(:,1:2) = subcondcentroid(c,1:2);
    bwd(n) =inter_ind_vs_state_distance(subcondcentroid_perm);
end

histogram(bwd,'EdgeAlpha',0);
y1=get(gca,'ylim');
hold on
plot([bwd0, bwd0],y1);
xlabel('Inter/Intra Individual Distance');
ylabel('Count');
p = sum(bwd0 < bwd) / n_perm;
mean_bw = mean(bwd);
std_bw = std(bwd);
disp(['permuted p = ', int2str(p)])

%% Figure 3E Clustering subjects
edist = squareform(pdist(cell2mat(ldspace(:,4:10)),'seuclidean'));
[edist_sub] = extract_dissimilar(edist, subidx, unique(ldspace(:,2)));
Z = linkage(edist_sub);
figure();
new_label = {'S1-1','S1-2','S2-1','S2-2','S3-1','S3-2','S4-1','S5-1','S6-1','S7-1',...
    'S7-2','S8-1','S9-1','S10-1','S10-2','S11-1','S12-1','S13-1','S14-1','S15-1'};
[H,T,OUTPERM] = dendrogram(Z, 'Labels',new_label);
for i=1:19
    H(i).Color = [0,0,0];
end
xtickangle(45);
ylabel('Latent EEG Distance');
xlabel('Participation');

% calculate distance within individual & between individual

pairlist = [1,2;3,4;5,6;10,11;14,15];
dist_within = [];
dist_between = [];
count=0
for s1 = 1:19
    for s2 = (s1+1):20
        count=count+1;
        if find(pairlist(:,1) == s1)
            if (s2 - s1) == 1
                dist_within(end+1) = edist_sub(s1,s2);
            else
                dist_between(end+1) = edist_sub(s1,s2);
            end          
        else
            dist_between(end+1) = edist_sub(s1,s2);
        end
    end
end
% stats
mean_dist_between = mean(dist_between);
std_dist_between = std(dist_between);
mean_dist_within = mean(dist_within);
std_dist_within = std(dist_within);
[h,p,ci,stats]=ttest2(dist_between,dist_within);
disp(['T(',int2str(stats.df),')=',num2str(stats.tstat),', p=',num2str(p)])

%% Figure 4: RSA analysis
edist = squareform(pdist(cell2mat(ldspace(:,4:10)),'seuclidean'));
load('mixed_model_all.mat');
behvars = behmat(:,4:10);
behdist = squareform(pdist(table2array(behvars), 'seuclidean'));
% correlate with edist
r = corr(uptriangle(behdist), uptriangle(edist));
scatter(uptriangle(edist),uptriangle(behdist), 3,'filled','MarkerFaceAlpha',.05)
ylabel('Behavioral distance');
xlabel('EEG distance');
title(['All: r = ', num2str(round(r,3))]);
lsline;
% permutation to assess the statistical significance, p
[r_news, r0, p] = permRSA(behdist, edist, 5000);

% if without the lower cluster
a = uptriangle(behdist);
b = uptriangle(edist);
mask_b = (b>1)
corr(a(mask_b),b(mask_b))
%% Figure 5A: within same individual vs. between different individual distance
pairlist = [1,2;3,4;5,6;10,11;14,15];

% calculate the actual distribution
dist_within = [];
dist_between = [];
for ld=1:7
    edist = squareform(pdist(cell2mat(ldspace(:,ld+3)),'euclidean'));
    [edist_sub] = extract_dissimilar(edist, subidx, unique(ldspace(:,2)));
    temp_within = [];
    temp_between = [];
    for s1 = 1:19
        for s2 = (s1+1):20
            if find(pairlist(:,1) == s1)
                if (s2 - s1) == 1
                    temp_within(end+1) = edist_sub(s1,s2);
                else
                    temp_between(end+1) = edist_sub(s1,s2);
                end          
            else
                temp_between(end+1) = edist_sub(s1,s2);
            end
        end
    end
    dist_within = [dist_within;temp_within];
    dist_between = [dist_between;temp_between];
end
ratio_sameld_all=[];
for ld=1:7
    k=0;
    for w=1:size(dist_within,2)
        for b=1:size(dist_between,2)
            k=k+1;
            ratio_sameld_all(ld,k) = dist_within(ld,w)/dist_between(ld,b);
        end
    end
end

% distribution of every single pair's within/between ratio
% do a bootstrap of 1000 times
ratio_null_all=[];
for sim=1:1000
    ld1 = randi([1 7],1);
    ld2 = randi([1 7],1);
    w = randi([1 size(dist_within,2)],1);
    b = randi([1 size(dist_between,2)],1);
    ratio_null_all(sim) = dist_within(ld1,w)/dist_between(ld2,b);
end


% test significance
stats = {};
for ld=1:7
    [h(ld),p(ld),ci,stats{ld}] = ttest2(ratio_null_all,ratio_sameld_all(ld,:));
end
p_individual_bonf = p*7;

% print the mean and sd of LD 1, 3, 5 and null distribution
for ld=1:7
    disp(['LD',num2str(ld),': Mean = ',num2str(mean(ratio_sameld_all(ld,:))),...
        ' STD = ',num2str(std(ratio_sameld_all(ld,:)))]);
end
% print the null distribution
disp(['null distribution: Mean = ',num2str(mean(ratio_null_all)),...
    ' STD = ',num2str(std(ratio_null_all))]);


figure;
bar(mean(ratio_sameld_all,2)');
yline(mean(ratio_null_all))
ylim([0 5]);
xlabel('LD');
ylabel('Repeated/Non-repeated participation distance ratio')


%% Figure 5B: anova on normalized LD to test inter-state (intra-individual) differences
% first normalize the LD
LD_norm = [];
for i=1:size(LD_table,1)
    subname_i = LD_table.participation(i);
    trial_i = LD_table.trial(i);
    idx_rest = find((LD_table.trial == trial_i) .* (strcmp(LD_table.participation,subname_i)));
    LD_norm(i,:) = table2array(LD_table(i,4:10)) - mean(table2array(LD_table(idx_rest,4:10)),1);
end
% ANOVA across conditions
p_cond = [];
for ld = [1,3,5]
    [p_cond(ld),~,stats] = anova1(LD_norm(:,ld),LD_table.condition);
    [c,m,h,gnames] = multcompare(stats);
    results{ld} = array2table([c,m],"RowNames",gnames, ...
    "VariableNames",["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value","Mean","Standard Error"]);
end
means = [mean(LD_norm(1:116,:),1);mean(LD_norm(117:232,:),1);mean(LD_norm(233:348,:),1)]';
stds = [std(LD_norm(1:116,:),[],1);std(LD_norm(117:232,:),[],1);std(LD_norm(233:348,:),[],1)];
% Bonforroni correction
p_cond_bonf = p_cond*7;



%% Figure 6: 2-stage tracing of the LD to PSD
% LD -> NMF done in Python and output saved
label_data = readtable('../data/PSD_label.csv');
load(['../data/LD_NMF_important.mat']);
heatmap(NMF_per_LD);

% find the NMF primary components to LD 1, 3, 5
for LD = [1,3,5]
    disp(['LD ',num2str(LD)]);
    for freq=3:4
        disp(['Freq ',num2str(freq)]);
        NMF_per_LD_freq = NMF_per_LD([40*(freq-1)+1:40*freq],:);
        disp(find(NMF_per_LD_freq(:,LD)))
    end
end

% NMF -> PSD
load(['../data/NMF_H.mat']);
load(['../data/NMF_scores.mat']);
load('bluewhiteorange.mat');
cmap = colormap(:,1:3);
clearvars colormap
% visualize beta NMF 1,2,3,8,11
H_freq_beta = reshape(H(3,:,:),40,128);
% take the pseudo inverse, which is how PSD channel would combine to NMF
NMF_weight_beta = pinv(H_freq_beta)';
for NMF = [1,2,3,8,11]
    figure
    plot_topography('all', NMF_weight_beta(NMF,:), 0,'biosemi128_corrected.mat');
    title(['Beta - NMF ',num2str(NMF)]);
    colormap(cmap);
    caxis([-0.12,0.12]);
end
% similarly do the low gamma
H_freq_lowgamma = reshape(H(4,:,:),40,128);
% take the pseudo inverse, which is how PSD channel would combine to NMF
NMF_weight_lowgamma = pinv(H_freq_lowgamma)';
for NMF = [3,30]
    figure
    plot_topography('all', NMF_weight_lowgamma(NMF,:), 0,'biosemi128_corrected.mat');
    title(['Low gamma - NMF ',num2str(NMF)]);
    colormap(cmap);
    caxis([-0.12,0.12]);
end

%% Figure 6-2: direct map from LD to PSD
load(['../data/LD_NMF_project.mat']);
load(['../data/NMF_H.mat']);
load('bluewhiteorange.mat');
cmap = colormap(:,1:3);
clearvars colormap
load('bwrcolormap.mat');


freqnames = {'Theta','Alpha','Beta','Low Gamma'};
% calculate channel weight on LD
LD_2_chan = {};
for ld = 1:7
    for freq = 1:4
        H_freq = reshape(H(freq,:,:),40,128);
        weight_NMF = pinv(H_freq);
        loading_freq = LD_loading([40*(freq-1)+1:40*freq],:);
        LD_2_chan{freq} = weight_NMF*loading_freq;
    end
end

% apply the weighted mask onto the PSD
load('../data/NormPSD.mat');
weighted_psd_norm_final = {};
for ld=1:7
    for freq = 1:4
        for cond=1:3
            weighted_psd_norm_final{ld,freq}(:,cond) = mean(psd_norm_final{freq+1}(condinx(:,cond),:),1)'.*LD_2_chan{freq}(:,ld);
        end
    end
end

% make all the plots
for ld=1:7
    for freq=1:4
        figure;
        subplot(1,4,1)
        plot_topography('all', LD_2_chan{freq}(:,ld)', 0,'/Users/wuqy0214/Documents/MATLAB/toolbox/plot_topography/plot_topography/biosemi128_corrected.mat');
        caxis([-2 2]);
        colormap(bwrcolormap);
        for cond=1:3
            subplot(1,4,cond+1)
            plot_topography('all', weighted_psd_norm_final{ld,freq}(:,cond),1,'biosemi128_corrected.mat');
            caxis([-1 1]) 
            colormap(cmap);
        end
    end
end
