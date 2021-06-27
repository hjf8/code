%采样频率
fs=5000;
nfft=10240;
%信号
fid=fopen('a.txt','r');%故障
N=512;
xdata=fscanf(fid,'%f',N);

fclose(fid);
xdata=(xdata-mean(xdata))/std(xdata,1);
%功率谱
figure(1);
subplot(221);
plot(xdata);
xlabel('时间 t/s');
ylabel('幅值 A/W');
Y=abs(fft(xdata,nfft));
subplot(222);
plot((0:nfft/2-1)/nfft*fs,Y(1:nfft/2));
xlabel('频率 f/Hz');
ylabel('功率谱 P/W');
%--------------------------------------------------------------------------
%5层小波包分解;
T=wpdec(xdata,3,'db1');
plot(T);%最后一句为显示小波包四层分解树结构

%-----------------------------------------------------------------------
figure(2);
subplot(211);
%重构第一层各系数
y10=wprcoef(T,[1,0]);
y11=wprcoef(T,[1,1]);
figure(2);
subplot(2,2,1);
plot(y10);
xlabel('第一层低频系数时间 t/s');
ylabel('第一层低频系数幅值 A/W');
Y10=abs(fft(y10,nfft));
subplot(2,2,2);
plot((0:nfft/2-1)/nfft*fs,Y10(1:nfft/2));
xlabel('第一层低频系数频率 f/Hz');
ylabel('第一层低频系数功率谱 P/W');
subplot(2,2,3);
plot(y11);
xlabel('第一层高频系数时间 t/s');
ylabel('第一层高频系数幅值 A/W');
Y11=abs(fft(y11,nfft));
subplot(2,2,4);
plot((0:nfft/2-1)/nfft*fs,Y11(1:nfft/2));
xlabel('第一层高频系数频率 f/Hz');
ylabel('第一层高频系数功率谱 P/W');
%------------------------------------------------------------------

%-----------------------------------------------------------------
%重构第二层各系数
y20=wprcoef(T,[2,0]);
y21=wprcoef(T,[2,1]);
y22=wprcoef(T,[2,2]);
y23=wprcoef(T,[2,3]);
figure(3);
subplot(4,2,1);
plot(y20);
xlabel('Y10分解的低频系数时间t/s');
ylabel('幅值A/W');
Y20=abs(fft(y20,nfft));
subplot(4,2,2);
plot((0:nfft/2-1)/nfft*fs,Y20(1:nfft/2));
xlabel('Y10分解的低频系数频率f/Hz');
ylabel('功率谱P/W');
subplot(4,2,3);
plot(y21);
xlabel('Y10分解的高频系数时间t/s');
ylabel('幅值A/W');
Y21=abs(fft(y21,nfft));
subplot(4,2,4);
plot((0:nfft/2-1)/nfft*fs,Y21(1:nfft/2));
xlabel('Y10分解的高频系数频率f/Hz');
ylabel('功率谱P/W');
subplot(4,2,5);
plot(y22);
xlabel('Y11分解的低频系数时间t/s');
ylabel('幅值A/W');
Y22=abs(fft(y22,nfft));
subplot(4,2,6);
plot((0:nfft/2-1)/nfft*fs,Y22(1:nfft/2));
xlabel('Y11分解的低频系数频率f/Hz');
ylabel('功率谱P/W');
subplot(4,2,7);
plot(y23);
xlabel('Y11分解的高频系数时间t/s');
ylabel('幅值A/W');
Y23=abs(fft(y23,nfft));
subplot(4,2,8);
plot((0:nfft/2-1)/nfft*fs,Y23(1:nfft/2));
xlabel('Y11分解的高频系数频率f/Hz');
ylabel('功率谱P/W');
%------------------------------------------------------------------

%-----------------------------------------------------------------
%重构第三层各系数
y30=wprcoef(T,[3,0]);
y31=wprcoef(T,[3,1]);
y32=wprcoef(T,[3,2]);
y33=wprcoef(T,[3,3]);
y34=wprcoef(T,[3,4]);
y35=wprcoef(T,[3,5]);
y36=wprcoef(T,[3,6]);
y37=wprcoef(T,[3,7]);
figure(4);
subplot(9,2,1);
plot(y30);
xlabel('Y20分解的低频系数时间t/s');
ylabel('幅值A/W');
Y30=abs(fft(y30,nfft));
subplot(9,2,2);
plot((0:nfft/2-1)/nfft*fs,Y30(1:nfft/2));
xlabel('Y20分解的低频系数频率f/Hz');
ylabel('功率谱P/W');
subplot(9,2,3);
plot(y31);
xlabel('Y20分解的高频系数时间t/s');
ylabel('幅值A/W');
Y31=abs(fft(y31,nfft));
subplot(9,2,4);
plot((0:nfft/2-1)/nfft*fs,Y31(1:nfft/2));
xlabel('Y20分解的高频系数频率f/Hz');
ylabel('功率谱P/W');
subplot(9,2,5);
plot(y32);
xlabel('Y21分解的低频系数时间t/s');
ylabel('幅值A/W');
Y32=abs(fft(y32,nfft));
subplot(9,2,6);
plot((0:nfft/2-1)/nfft*fs,Y32(1:nfft/2));
xlabel('Y21分解的低频系数时间t/s');
ylabel('功率谱P/W');
subplot(9,2,7);
plot(y33);
xlabel('Y21分解的高频系数时间t/s');
ylabel('幅值A/W');
Y33=abs(fft(y33,nfft));
subplot(9,2,8);
plot((0:nfft/2-1)/nfft*fs,Y33(1:nfft/2));
xlabel('Y21分解的高频系数时间t/s');
ylabel('功率谱P/W');
subplot(9,2,9);
plot(y34);
xlabel('Y22分解的低频系数时间t/s');
ylabel('幅值A/W');
Y34=abs(fft(y34,nfft));
subplot(9,2,10);
plot((0:nfft/2-1)/nfft*fs,Y34(1:nfft/2));
xlabel('Y22分解的低频系数时间t/s');
ylabel('功率谱P/W');
subplot(9,2,11);
plot(y35);
xlabel('Y22分解的高频系数时间t/s');
ylabel('幅值A/W');
Y35=abs(fft(y35,nfft));
subplot(9,2,12);
plot((0:nfft/2-1)/nfft*fs,Y35(1:nfft/2));
xlabel('Y22分解的高频系数时间t/s');
ylabel('功率谱P/W');
subplot(9,2,13);
plot(y36);
xlabel('Y23分解的低频系数时间t/s');
ylabel('幅值A/W');
Y36=abs(fft(y36,nfft));
subplot(9,2,14);
plot((0:nfft/2-1)/nfft*fs,Y36(1:nfft/2));
xlabel('Y23分解的低频系数时间t/s');
ylabel('功率谱P/W');
subplot(9,2,15);
plot(y37);
xlabel('Y23分解的高频系数时间t/s');
ylabel('幅值A/W');
Y37=abs(fft(y37,nfft));
subplot(9,2,16);
plot((0:nfft/2-1)/nfft*fs,Y37(1:nfft/2));
xlabel('Y23分解的低频系数时间t/s');
ylabel('功率谱P/W');
%------------------------------------------------------------------

%-----------------------------------------------------------------













































