clear all, close all

x=rand(20,1)*10-5;
y=x.*sin(x);
x=[x;7.5;8;10.5;11];
y=[y;7;-7;7;-7];

plot(x, y, '+');

ell = 1; sn = 1; sf = 1; % homoscedastic_GP_reg_large_observation_noise
% ell = 1; sn = 0; sf = 1; % homoscedastic_GP_reg_small_observation_noise

meanfunc = @meanConst; hyp.mean = [0.];
covfunc = @covSEiso; hyp.cov = log([ell; sf]);
likfunc = @likGauss; hyp.lik = log(sn);
 
nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y)

z = linspace(-6, 11, 101)';
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; 
plot(z, m); 
scatter(x, y);
ylim([-10,10]);
