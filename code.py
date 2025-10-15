!pip install networkx matplotlib numpy gymnasium opencv-python --quiet

import random, itertools, time
import numpy as np
import networkx as nx
import cv2
from tqdm import tqdm

# ---------------------------
# Environment
# ---------------------------
class SpaceEnv:
    def __init__(self, G, sample_nodes, base_node="Base", max_steps=60, fuel_capacity=40, seed=42):
        self.G = G
        self.nodes = list(G.nodes())
        self.sample_nodes = sample_nodes
        self.base_node = base_node
        self.max_steps = max_steps
        self.fuel_capacity = fuel_capacity
        self.seed = seed
        self.reset()

    def reset(self):
        self.current = self.base_node
        self.collected = [0]*len(self.sample_nodes)
        self.steps = 0
        self.time_remaining = self.max_steps
        self.fuel = self.fuel_capacity
        return self._get_obs()

    def _get_obs(self):
        node_idx = self.nodes.index(self.current)
        node_one_hot = np.zeros(len(self.nodes))
        node_one_hot[node_idx] = 1
        mask = np.array(self.collected)
        tnorm = np.array([self.time_remaining/self.max_steps])
        fnorm = np.array([self.fuel/self.fuel_capacity])
        return np.concatenate([node_one_hot, mask, tnorm, fnorm])

    def step(self, action):
        reward=0; done=False
        self.steps+=1; self.time_remaining-=1; self.fuel-=1
        if action<len(self.nodes):
            target=self.nodes[action]
            if target==self.current:
                reward-=1
            elif self.G.has_edge(self.current,target):
                edge=self.G[self.current][target]
                reward-=edge.get('cost',1)
                if random.random()<edge.get('risk',0.0):
                    reward-=20; done=True
                else: self.current=target
            else: reward-=5
        elif action==len(self.nodes):
            if self.current in self.sample_nodes:
                idx=self.sample_nodes.index(self.current)
                if self.collected[idx]==0: self.collected[idx]=1; reward+=15
                else: reward-=1
            else: reward-=1
        elif action==len(self.nodes)+1:
            if self.current==self.base_node:
                if all(self.collected): reward+=30; done=True
                else: reward-=2
            elif self.G.has_edge(self.current,self.base_node):
                edge=self.G[self.current][self.base_node]
                reward-=edge.get('cost',1)
                if random.random()<edge.get('risk',0.0): reward-=20; done=True
                else: self.current=self.base_node
            else: reward-=5
        if self.time_remaining<=0 or self.fuel<=0: reward-=10; done=True
        return self._get_obs(), reward, done

# ---------------------------
# Galaxy creation
# ---------------------------
def create_sample_galaxy():
    G=nx.Graph()
    nodes={"Base":(0,0),"A":(2,1),"B":(3,-1),"C":(5,0),"D":(4,3),"E":(7,2)}
    for n,pos in nodes.items(): G.add_node(n,pos=pos)
    edges=[("Base","A",2,0.02),("A","B",3,0.05),("B","C",2,0.12),
           ("A","D",4,0.1),("D","E",3,0.15),("C","E",2,0.2),("C","D",3,0.05)]
    for u,v,cost,risk in edges: G.add_edge(u,v,cost=cost,risk=risk)
    samples=["B","E"]
    return G,samples,"Base"

# ---------------------------
# A* multi-goal planner
# ---------------------------
def astar_multi_goal_plan(G,base,samples):
    import heapq
    start_mask=tuple(0 for _ in samples)
    start=(base,start_mask)
    def heuristic(node,mask):
        remaining=[s for i,s in enumerate(samples) if mask[i]==0]
        total=0; cur=node
        while remaining:
            dists=[(nx.shortest_path_length(G,cur,r,weight='cost'),r) for r in remaining]
            dists.sort(); d,r=dists[0]; total+=d; cur=r; remaining.remove(r)
        if cur!=base: total+=nx.shortest_path_length(G,cur,base,weight='cost')
        return total
    open_heap=[]; gscore={start:0}; fscore={start:heuristic(base,start_mask)}
    heapq.heappush(open_heap,(fscore[start],start)); parent={start:None}
    while open_heap:
        _,current=heapq.heappop(open_heap)
        cur_node,mask=current
        if all(mask) and cur_node==base:
            path=[]; s=current
            while s is not None: path.append(s[0]); s=parent[s]
            path.reverse(); return path
        for nb in G.neighbors(cur_node):
            new_mask=list(mask)
            if nb in samples: new_mask[samples.index(nb)]=1
            new_mask=tuple(new_mask); tentative_g=gscore[current]+G[cur_node][nb]['cost']
            neighbor=(nb,new_mask)
            if tentative_g<gscore.get(neighbor,1e9):
                parent[neighbor]=current; gscore[neighbor]=tentative_g
                fscore[neighbor]=tentative_g+heuristic(nb,new_mask)
                heapq.heappush(open_heap,(fscore[neighbor],neighbor))
    return None

# ---------------------------
# Q-Learning agent
# ---------------------------
class QLearner:
    def __init__(self,env,lr=0.2,gamma=0.98,eps=0.3,eps_decay=0.995):
        self.env=env; self.lr=lr; self.gamma=gamma; self.eps=eps; self.eps_decay=eps_decay
        self.nodes=env.nodes; self.samples=env.sample_nodes
        self.state_map={}; self.rev_state={}; self._build_state_space()
        self.Q=np.zeros((len(self.state_map),env.nodes.__len__()+2))
    def _build_state_space(self):
        fuel_buckets=6; idx=0
        for cur in self.nodes:
            for mask in itertools.product([0,1],repeat=len(self.samples)):
                for fb in range(fuel_buckets):
                    self.state_map[(cur,mask,fb)]=idx; self.rev_state[idx]=(cur,mask,fb); idx+=1
    def _encode(self,obs):
        node_idx=np.argmax(obs[:len(self.nodes)])
        mask=tuple(int(x) for x in obs[len(self.nodes):len(self.nodes)+len(self.samples)])
        fb=int(obs[-1]*5)
        return self.state_map.get((self.nodes[node_idx],mask,fb),0)
    def choose(self,s): return np.random.randint(0,self.Q.shape[1]) if np.random.rand()<self.eps else int(np.argmax(self.Q[s]))
    def train(self,episodes=500,max_steps=60):
        for ep in range(episodes):
            obs=self.env.reset(); s=self._encode(obs)
            for _ in range(max_steps):
                a=self.choose(s); obs,r,done=self.env.step(a); s2=self._encode(obs)
                self.Q[s,a]+=self.lr*(r+self.gamma*np.max(self.Q[s2])-self.Q[s,a]); s=s2
                if done: break
    def policy(self,obs): return int(np.argmax(self.Q[self._encode(obs)]))

# ---------------------------
# Video animation
# ---------------------------
def animate_video(G,astar_path,ql_path,filename="spaceship_sim.mp4",H=400,W=600):
    pos=nx.get_node_attributes(G,'pos')
    scaled={n:(int(x/W*500)+50,int(y/H*300)+50) for n,(x,y) in pos.items()}
    steps_per_edge=15
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(filename,fourcc,10,(W,H))
    t=0; idx_astar=0; idx_q=0; done_a=False; done_q=False
    while not (done_a and done_q):
        img=np.zeros((H,W,3),dtype=np.uint8); img[:,:,:]=20
        # edges
        for u,v in G.edges(): cv2.line(img,scaled[u],scaled[v],(100,100,100),2)
        # nodes
        for n,(x,y) in scaled.items():
            color=(0,255,255) if n=='Base' else (0,200,0) if n in ["B","E"] else (200,200,255)
            cv2.circle(img,(x,y),20,color,-1); cv2.putText(img,n,(x-10,y+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        # A* spaceship
        if not done_a:
            start=scaled[astar_path[idx_astar]]; end=scaled[astar_path[idx_astar+1]] if idx_astar+1<len(astar_path) else start
            alpha=t/steps_per_edge; cx=int(start[0]*(1-alpha)+end[0]*alpha); cy=int(start[1]*(1-alpha)+end[1]*alpha)
            cv2.circle(img,(cx,cy),10,(0,255,255),-1); cv2.putText(img,"A*",(cx+10,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        # Q spaceship
        if not done_q:
            start=scaled[ql_path[idx_q]]; end=scaled[ql_path[idx_q+1]] if idx_q+1<len(ql_path) else start
            alpha=t/steps_per_edge; cx2=int(start[0]*(1-alpha)+end[0]*alpha); cy2=int(start[1]*(1-alpha)+end[1]*alpha)
            cv2.circle(img,(cx2,cy2),10,(255,0,0),-1); cv2.putText(img,"RL",(cx2+10,cy2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        out.write(img)
        t+=1
        if t>steps_per_edge: t=0; idx_astar+=1 if not done_a else 0; idx_q+=1 if not done_q else 0
        if idx_astar>=len(astar_path)-1: done_a=True
        if idx_q>=len(ql_path)-1: done_q=True
    out.release()
    print(f"Video saved as {filename}")

# ---------------------------
# Main
# ---------------------------
def main():
    G,samples,base=create_sample_galaxy()
    astar_path=astar_multi_goal_plan(G,base,samples)
    env=SpaceEnv(G,samples,base)
    q_agent=QLearner(env); q_agent.train(episodes=500)
    obs=env.reset(); q_path=[env.current]; done=False
    while not done:
        a=q_agent.policy(obs); obs,_,done=env.step(a); q_path.append(env.current)
    animate_video(G,astar_path,q_path,"spaceship_sim.mp4")
    print("Download the video from the Colab file explorer.")

if __name__=="__main__":
    main()
