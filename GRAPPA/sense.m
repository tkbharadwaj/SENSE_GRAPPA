load('raw_data.mat');
[Nx,Ny,Nz,Nc] = size(raw);

data = ifft2c(raw);

imgs = abs(data);

N = ndims(imgs);
sz = size(imgs,N);
n = ceil(sqrt(sz));
m = ceil(sz/n);
idx = repmat({':'},1,N);


load('img_patref.mat')
disp(size(img_patref))
%mm = coil_estimation_ESPIRIT(img_patref, 24, [6, 6], 0.8);
%show_grid(abs(mm(:, :, 1, 1:16)), [0 1], jet)

show_img(abs(coil_combine(data)))
show_grid(abs(imgs))
d  = coil_estimation(data);
mm = coil_estimation_ESPIRIT(data, 24, [6, 6], 0.8);
d = mm;
show_grid(abs(mm), [0 1], jet);
show_grid(abs(d), [0 1], jet);
raw_undersampled = undersample(raw, 4)
und_imgs = ifft2c(raw_undersampled)
show_grid(abs(und_imgs))
recon = SENSE(und_imgs, d, 4)
show_img(abs(recon), [0 16], gray)


function map=coil_estimation_ESPIRIT(raw_data, num_acs, kernel_size, eigen_thresh)
    

    img_patref = raw_data;
    receive = zeross(size(img_patref));
    tic
    parfor slc_select = 1:s(img_patref,3)     
        disp(num2str(slc_select))
        
        [maps, weights] = ecalib_soft( fft2c( sq(img_patref(:,:,slc_select,:)) ), num_acs, kernel_size, eigen_thresh );
    
        receive(:,:,slc_select,:) = permute(dot_mult(maps, weights >= eigen_thresh ), [1,2,4,3]);
    end 
    toc
    map = receive;

end

function img = coil_combine(imgs)
img  = sqrt(sum(abs(imgs).^2,4));
end

function sens = coil_estimation(imgs)
    [Nx,Ny,Nz,Nc] = size(imgs);
    m_aprox  = sqrt(sum(abs(imgs).^2,4));
    Cs1 = bsxfun(@rdivide, imgs, m_aprox);
    kern = ones(9) * (1/81);
    for i = 1:Nc
        Cs1(:, :, 1, i) = conv2( Cs1(:, :, 1, i), kern, "same");
    end
    threshold = 0.05*max(m_aprox(:));
    mask = abs(m_aprox) > threshold;
    mask = medfilt2(mask, [11, 11]);
    Cs2 = Cs1.*mask;
    sens = Cs2;
end


function recon = SENSE(imgs, sens, accel_factor)
    
    [Nx,Ny,Nz,Nc] = size(imgs);
    recon = zeros(Nx, Ny); 
    if accel_factor == 2
        for i = 1:Nx/2
            for j = 1:Ny
                C =  transpose(squeeze([sens(i, j, 1, :); sens(i + Nx/2, j, 1, :)])) ; 
                y = transpose(transpose ( squeeze(imgs(i, j, 1, :) )));
                x = pinv(C)*y;
                recon(i, j) = x(1);
                recon(i + Nx/2, j) = x(2);
            end
        end
    
    else
        for i = 1:Nx/accel_factor
            for j = 1:Ny
                C = [sens(i, j, 1, :)];
                for jj = 1:accel_factor-1
                    C = [C; sens(i + (Nx/accel_factor)*jj, j, 1, :)] ;                   
                end
                C = transpose(squeeze(C));
                
                y = transpose(transpose ( squeeze(imgs(i, j, 1, :) )));
                x = pinv(C)*y;

                for jj = 0:accel_factor-1
                    recon(i + jj*Nx/accel_factor, j) = x(jj+1);
                end

            end
        end
    end
    
end



function undersampled = undersample(kspace, accel_factor)
undersampled = zeros(size(kspace));
for i = 1:accel_factor:size(kspace, 1)-1
    undersampled(i, :, :, : ) = kspace(i, :, :, : );
end

end
%helper functions taken from https://github.com/mchiew/SENSE-tutorial/blob/main/SENSE_tutorial.m
function show_grid(data, cscale, cmap)
    if nargin < 2
        cscale = [];
    end
    if nargin < 3
        cmap = gray;
    end
    figure();
    N = ndims(data);
    sz = size(data,N);
    n = ceil(sqrt(sz));
    m = ceil(sz/n);
    idx = repmat({':'},1,N);
    for i = 1:m
        for j = 1:n
            idx{N} = (i-1)*m+j;
            subplot('position',[(i-1)/m (n-j)/n (1/m-0.005) (1/n-0.005)]);
            imshow(data(idx{:}),cscale,'colormap',cmap);
        end
    end
end

function show_img(data, cscale, cmap)
   if nargin < 2 || isempty(cscale)
       cscale = [-inf inf];
   end
   if nargin < 3
       cmap = gray;
   end
   figure();
   imagesc(data);
   axis equal
   colormap(cmap);
   clim(cscale);
   plotH = gca;
   plotH.XTick = [];plotH.YTick = [];plotH.YColor = 'w';plotH.XColor = 'w';
end

%function out = fft2c(input)
%    out = fftshift(fft(ifftshift(input,1),[],1),1);
%    out = fftshift(fft(ifftshift(out,2),[],2),2);
%end

function out = ifft2c(input)
    out = fftshift(ifft(ifftshift(input,1),[],1),1);
    out = fftshift(ifft(ifftshift(out,2),[],2),2);
end