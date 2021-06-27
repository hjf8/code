filepath='TRAIN\';%文件夹的路径
   for i=101:104  %n是要读入的文件的个数

        load([filepath 'TRAIN' num2str(i) '.mat'])
        dataname=['d0' num2str(i) '.dat']
        chr=[filepath dataname]
        d0=load(chr)
        figure;
        plot(1:length(d0),d0);
        clear(chr)
        clear(dataname)
   end