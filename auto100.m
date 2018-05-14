
f1=figure(); %shows original image
f2 = figure();  %shows reconstructed image of autenc1
f3 = figure();  %confusion plot of shows reconstructed image of autenc2
f4= figure(); %confusion plot of testing data with deepnet
hiddenSize1 = 300;
hiddenSize2 = 300;
s1=10;   % image resize
ep=10;  % number of epochs
imageWidth = s1;
imageHeight = s1;
inputSize = imageWidth*imageHeight;
unzip('home1.zip');
images = imageDatastore('home1','IncludeSubfolders',true,'LabelSource','foldernames');
images.ReadFcn = @(loc)imresize(imread(loc),[s1,s1]);
images.Labels(1);
length(images.Labels);
[trainImages,valImages] = splitEachLabel(images,0.7,'randomized');

F=[];   %training data targets
for c = 1:length(trainImages.Labels)
  if trainImages.Labels(c) == "daisy's"
      t=[1 0 0 0 0]' ; 
  end
  if trainImages.Labels(c) == "dandelions"
      t=[0 1 0 0 0]';
      end
      if trainImages.Labels(c) == "roses"
      t=[0 0 1 0 0]'; 
      end
      if trainImages.Labels(c) == "sunflowers"
      t=[0 0 0 1 0]' ;  
      end
      if trainImages.Labels(c) == "tulips"
      t=[0 0 0 0 1]';
      end
  F = [F,t];
end

Z = readall(trainImages);
for z=1:length(trainImages.Labels)
    Z{z} =im2double(Z{z});
end
Z = Z';

G=[];    %testing/validation  target data
for k = 1:length(valImages.Labels)
  if valImages.Labels(k) == "daisy's"
      s=[1 0 0 0 0]' ; 
      e=e+1;
  end
  if valImages.Labels(k) == "dandelions"
      s=[0 1 0 0 0]';
      end
      if valImages.Labels(k) == "roses"
      s=[0 0 1 0 0]'; 
      end
      if valImages.Labels(k) == "sunflowers"
      s=[0 0 0 1 0]' ;  
      end
      if valImages.Labels(k) == "tulips"
      s=[0 0 0 0 1]';
      end
  G = [G,s];
end

Y = readall(valImages); %testing/validation data
for w=1:length(valImages.Labels)
    Y{w} =im2double(Y{w});
end
Y = Y';

xTrn = zeros(inputSize*3,numel(trainImages.Labels));  %squash training
for i = 1:numel(trainImages.Labels)
   xTrn(:,i) = Z{i}(:);
end

xTst = zeros(inputSize*3,length(valImages.Labels));   %squash testing
for i = 1:length(valImages.Labels)
    xTst(:,i) = Y{i}(:);
end

rng('default')
autoenc1 = trainAutoencoder(Z ,hiddenSize1, ...
    'MaxEpochs',ep, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

xReconstructed1 = predict(autoenc1,Z);

figure(f1);  %orig image display
for u = 1:25
    subplot(5,5,u);
    imshow(Z{u});
end

figure(f2);   %reconstruc image display
for u1 = 1:25
    subplot(5,5,u1);
    imshow(xReconstructed1{u1});
end

%xrec1 are autoenc1 reconstructed images to be classified
xrec1 = zeros(inputSize*3,length(trainImages.Labels));
for p = 1:length(trainImages.Labels)
    xrec1(:,p) = xReconstructed1{p}(:);
end

feat1 = encode(autoenc1,Z);

autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
 'MaxEpochs',ep, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(feat2,F,'MaxEpochs',ep);  %change feat2
deepnet = stack(autoenc1,autoenc2,softnet);

h=deepnet(xrec1);  %classify the autoenc1 reconstructed images
figure(f3);
plotconfusion(F,h);

deepnet = train(deepnet,xTrn,F);  %fine tune on training data
h2 = deepnet(xTst);  % classify with testing data
figure(f4);
plotconfusion(G,h2);

