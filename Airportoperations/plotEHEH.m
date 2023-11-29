load('EHEH_digraph.mat');
load('EHEH_serviceRoadDigraph.mat');
A = adjacency(G,'weighted');
B = incidence(G);

GX = graph(A+A');

GX.Nodes = G.Nodes;
GX.Nodes.z = 20*ones(70,1);
GS.Nodes.z = zeros(81,1);

gsplot = plot(GS,'XData',GS.Nodes.x,'YData',GS.Nodes.y,'NodeColor',[.7 .7 .7],"EdgeColor",[.7 .7 .7],'LineStyle','--','Interpreter','latex');
hold on
gplot = plot(GX,'XData',GX.Nodes.x,'YData',GX.Nodes.y,'NodeColor','k',"EdgeColor",'k','Interpreter','latex');
highlight(gplot,G.Nodes.gate==1,"NodeColor",'g');
highlight(gsplot,GS.Nodes.gate==1,"NodeColor",'g');
highlight(gplot,G.Nodes.runway==1,"NodeColor",'b');
%axis equal
set(gca,'TickLabelInterpreter','latex');
legend('$G_\mathrm{S}$','$G_\mathrm{X}$','Interpreter','latex','Location','northwest')