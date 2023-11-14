%% For convinience, we assume the order of the tensor is always 3;
clear;
addpath('tSVD','proxFunctions','solvers','twist');
addpath('ClusteringMeasure', 'LRR', 'Nuclear_norm_l21_Algorithm', 'unlocbox');

%load('yale.mat');
%load('my_extended_yaleB.mat');
%load('my_NH_face.mat');  % lambda = 10 bestComb:X3,X2
%load('YaleB_first10.mat'); % lambda = 0.001, S = Z1+Z2
load('my_ORL.mat'); % bestComb:X3,X2,X1 lambda = 100
%load('my_new_COIL20MV.mat');  %best combination X2, X3
% COIL_NUM = 1440;
% X1 = X1(:,1:COIL_NUM);
% X2 = X2(:,1:COIL_NUM);
% X3 = X3(:,1:COIL_NUM);
% gt = gt(1:COIL_NUM);
cls_num = length(unique(gt));
%% Note: each column is an sample (same as in LRR)
%% 
%data preparation...
 X{1} = X3; X{2} = X2; X{3} = X1;
 for v=1:3
    [X{v}]=NormalizeData(X{v});
     %X{v} = zscore(X{v},1);
end
% Initialize...

K = length(X); N = size(X{1},2); %sample number

for k=1:K
    Z{k} = zeros(N,N); %Z{2} = zeros(N,N);
    W{k} = zeros(N,N);
    G{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N); %E{2} = zeros(size(X{k},1),N);
    Y{k} = zeros(size(X{k},1),N); %Y{2} = zeros(size(X{k},1),N);
end

w = zeros(N*N*K,1);
g = zeros(N*N*K,1);
dim1 = N;dim2 = N;dim3 = K;
myNorm = 'tSVD_1';
sX = [N, N, K];
%set Default
parOP         =    false;
ABSTOL        =    1e-6;
RELTOL        =    1e-4;


Isconverg = 0;epson = 1e-7;
lambda = 0.2; %1.5 best
iter = 0;
mu = 10e-5; max_mu = 10e10; pho_mu = 2;
rho = 0.0001; max_rho = 10e12; pho_rho = 2;
tic;

while(Isconverg == 0)
    fprintf('----processing iter %d--------\n', iter+1);
    for k=1:K
        %1 update Z^k
        
        %Z{k}=inv(eye(N,N)+X{k}'*X{k})*(X{k}'*X{k} - X{k}'*E{k}+ F3_inv(G_bar){k}
        %                               + (X{k}'*Y{k} - W{k})/\mu);
        tmp = (X{k}'*Y{k} + mu*X{k}'*X{k} - mu*X{k}'*E{k} - W{k})./rho +  G{k};
        Z{k}=inv(eye(N,N)+ (mu/rho)*X{k}'*X{k})*tmp;
        
        %2 update E^k
        %F = [X{1}-X{1}*Z{1}+Y{1};X{2}-X{2}*Z{2}+Y{2}];
        %F = [X{1}-X{1}*Z{1};X{2}-X{2}*Z{2}];
        %F = [X{1}-X{1}*Z{1};X{2}-X{2}*Z{2};X{3}-X{3}*Z{3}];
        F = [X{1}-X{1}*Z{1}+Y{1}/mu;X{2}-X{2}*Z{2}+Y{2}/mu;X{3}-X{3}*Z{3}+Y{3}/mu];
        %F = [X{1}-X{1}*Z{1}+Y{1}/mu;X{2}-X{2}*Z{2}+Y{2}/mu];
        [Econcat] = solve_l1l2(F,lambda/mu);
        %F = F';
        %[Econcat,info] = prox_l21(F, 0.5/1);
        E{1} = Econcat(1:size(X{1},1),:);
        E{2} = Econcat(size(X{1},1)+1:size(X{1},1)+size(X{2},1),:);
        E{3} = Econcat(size(X{1},1)+size(X{2},1)+1:end,:);
        %3 update Yk
        %Y{k} = Y{k} + mu*(X{k}-X{k}*Z{k}-E{k});
        Y{k} = Y{k} + mu*(X{k}-X{k}*Z{k}-E{k});
    end
    
    %4 update G
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});
    z = Z_tensor(:);
    w = W_tensor(:);
    
    %twist-version
   [g, objV] = wshrinkObj(z + 1/rho*w,1/rho,sX,0,3)   ;
%    [g, objV] = shrinkObj(z + (1/rho)*w,...
%                         1/rho,myNorm,sX,parOP);
    G_tensor = reshape(g, sX);
    
    %5 update W
    w = w + rho*(z - g);
    
    %record the iteration information
    history.objval(iter+1)   =  objV;

    %% coverge condition
    Isconverg = 1;
    for k=1:K
        if (norm(X{k}-X{k}*Z{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}-X{k}*Z{k}-E{k},inf);
            fprintf('    norm_Z %7.10f    ', history.norm_Z);
            Isconverg = 0;
        end
        
        G{k} = G_tensor(:,:,k);
        W_tensor = reshape(w, sX);
        W{k} = W_tensor(:,:,k);
        if (norm(Z{k}-G{k},inf)>epson)
            history.norm_Z_G = norm(Z{k}-G{k},inf);
            fprintf('norm_Z_G %7.10f    \n', history.norm_Z_G);
            Isconverg = 0;
        end
    end
   
    if (iter>200)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
end
S = 0;
for k=1:K
    S = S + abs(Z{k})+abs(Z{k}');
end
% figure(1); imagesc(S);
% S_bar = CLR(S, cls_num, 0, 0 );
% figure(2); imagesc(S_bar);
C = SpectralClustering(S,cls_num);
[A nmi avgent] = compute_nmi(gt,C);
%C = SpectralClustering(abs(Z{1})+abs(Z{1}'),cls_num);
%[A nmi avgent] = compute_nmi(gt,C)
% C = SpectralClustering(abs(Z{2})+abs(Z{2}'),cls_num);
% [A nmi avgent] = compute_nmi(gt,C)
% C = SpectralClustering(abs(Z{3})+abs(Z{3}'),cls_num);
% [A nmi avgent] = compute_nmi(gt,C)
ACC = Accuracy(C,double(gt));
[f,p,r] = compute_f(gt,C);
[AR,RI,MI,HI]=RandIndex(gt,C);
toc;
%save('my_new_COIL20MV_res.mat','S','ACC','nmi','AR','f','p','r');

