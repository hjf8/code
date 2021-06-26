clear all
tic;
global k f
% for k=1:10
    f=116.8;
%ODE-solver options
%Type 'help odeset' in the Matlab window for more details
opts = odeset('RelTol',1e-8,'AbsTol',1e-8,'MaxStep',0.1);
% Initial condition
y0=[4.35,5.16,3.3185,4.45,0];
% y0=[1.3,3.1,5.0,-4.1,0];
% y0=[0.35,0.18,0.15,0.1,0];
% Integration time
T=20;
tspan = [0,T];
% Output is the time, states and events as in Matlab's standard output
tic;
[t,y]=ode45('xuanbiliang', tspan, y0,opts);
time=toc
n=90*round(length(t)/100);
t1=t(n:end);
y1=y(n:end,:);
N=length(t);
j=0;
% for i=2:N
%     %tmp=y1(1,5)+j*2*pi/2(此处适当调整)-pi/6（此处适当调整）;
%     tmp=y1(1,5)+j*2*pi/2-pi/2;
%     if y1(i,5)>=tmp
%         if abs(y1(i,5)-tmp)>abs(y1(i-1,5)-tmp)
%         figure(6),hold on       
%         set(gcf,'unit','normalized','position',[0,0.04,0.6,0.8]);%窗口尺寸
%         plot(y1(i-1,1),y1(i-1,2),'k.','MarkerSize',15,'linewidth',2);
%         set(gca,'linewidth',5);%坐标线粗1.5磅?
%         set(gca,'fontsize',40);
%         %plot(y1(i-1,3),y1(i-1,4),'k.');
%         else
%             %----------一阶庞加莱图----------
%             figure(6),hold on
%             set(gcf,'unit','normalized','position',[0,0.04,0.6,0.8]);%窗口尺寸
%             plot(y1(i,1),y1(i,2),'k.','MarkerSize',15,'linewidth',2);
%             set(gca,'linewidth',5);%坐标线粗1.5磅?
%             set(gca,'fontsize',40);
%             %plot(y1(i,3),y1(i,4),'k.');
%             saveas(gcf,'6','bmp')
%         end
%         j=j+1;
%     end
% end
%----------一阶相图----------
figure(1)
set(gca, 'Fontname', 'Times newman');
set(gcf,'unit','normalized','position',[0,0.04,0.6,0.8]);%窗口尺寸
plot(y(n:end,1),y(n:end,2),'MarkerSize',15,'linewidth',2)
xlabel('x1'),ylabel('x2')
set(get(gca,'XLabel'),'FontSize',55);
set(get(gca,'YLabel'),'FontSize',55);
set(get(gca,'ZLabel'),'FontSize',55);
set(gca,'linewidth',5);%坐标线粗1.5磅?
set(gca,'fontsize',40);
% saveas(gcf,'1','bmp')
% ----------一阶波形图----------
figure(2)
% set(gcf,'unit','normalized','position',[0,0.04,0.6,0.8]);%窗口尺寸
plot(t(n:end),y(n:end,1),'MarkerSize',15,'linewidth',2)
axis([990 1000 -inf inf])
xlabel('t'),ylabel('x1')
set(get(gca,'XLabel'),'FontSize',55);
set(get(gca,'YLabel'),'FontSize',55);
set(get(gca,'ZLabel'),'FontSize',55);
set(gca,'linewidth',5);%坐标线粗1.5磅?
set(gca,'fontsize',40);
% ---------二阶相图----------
figure(3)
% set(gcf,'unit','normalized','position',[0,0.04,0.6,0.8]);%窗口尺寸
plot(y(n:end,3),y(n:end,4),'linewidth',2)
xlabel('x3'),ylabel('x4')
set(get(gca,'XLabel'),'FontSize',55);
set(get(gca,'YLabel'),'FontSize',55);
set(get(gca,'ZLabel'),'FontSize',55);
set(gca,'linewidth',5);%坐标线粗1.5磅?
set(gca,'fontsize',40);
% ----------二阶波形图----------
figure(4)
% set(gcf,'unit','normalized','position',[0,0.04,0.6,0.8]);%窗口尺寸
plot(t(n:end),y(n:end,3),'linewidth',2)
axis([990 1000 -inf inf])
xlabel('t'),ylabel('x3')
set(get(gca,'XLabel'),'FontSize',55);
set(get(gca,'YLabel'),'FontSize',55);
set(get(gca,'ZLabel'),'FontSize',55);
set(gca,'linewidth',5);%坐标线粗1.5磅?
set(gca,'fontsize',40);
% ----------三维相图----------
figure(5)
% set(gcf,'unit','normalized','position',[0,0.04,0.6,0.8]);%窗口尺寸
plot3(y(n:end,1),y(n:end,2),y(n:end,3),'linewidth',2)
% plot3(y(1:end,1),y(1:end,2),y(1:end,3),'linewidth',2)
xlabel('x1'),ylabel('x2'),zlabel('x3')
set(get(gca,'XLabel'),'FontSize',55);
set(get(gca,'YLabel'),'FontSize',55);
set(get(gca,'ZLabel'),'FontSize',55);
set(gca,'linewidth',5);%坐标线粗1.5磅?
set(gca,'fontsize',40);
% end
toc