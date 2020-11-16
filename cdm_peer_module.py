# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#imports
import pandas as pd
import networkx as nx
import pickle
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt

#helper methods
def pickle_store(filename, ds_name):
    outfile = open(filename,'wb')
    pickle.dump(ds_name, outfile)
    outfile.close()

def pickle_restore(filename):
    infile = open(filename, 'rb')
    dsname = pickle.load(infile)
    infile.close()
    return dsname


# %%
#fetch all rawdata and save as pickle format for speed.
email_raw_df = pd.read_csv("r4.2/email.csv") # 2629979 rows, 11 columns
#index email data by date
email_raw_df['date'] = pd.to_datetime(email_raw_df['date'])
email_raw_df.set_index('date', inplace=True)

#get ldapdata
ldap_raw_9_12_df = pd.read_csv("r4.2/LDAP/2009-12.csv") # 2629979 rows, 11 columns

#pickle data for future use.
pickle_store("email_file", email_raw_df)
pickle_store("ldap_file", ldap_raw_9_12_df)


# %%

email_raw_df = pickle_restore("email_file")
# email_raw_df = email_raw_df['2010-01-01':'2010-03-1']
ldap_raw_9_12_df = pickle_restore("ldap_file")

ldap_uid_email_df = ldap_raw_9_12_df[['user_id', 'email']]
eid_email_map = ldap_uid_email_df.set_index('user_id').T.to_dict('records')[0]
pickle_store("eid_email_map_file", eid_email_map)

email_eid_map = ldap_uid_email_df.set_index('email').T.to_dict('records')[0]
pickle_store("email_eid_map_file", email_eid_map)

ldap_uid_role_df = ldap_raw_9_12_df[['user_id', 'role']]
eid_role_map = ldap_uid_role_df.set_index('user_id').T.to_dict('records')[0]
pickle_store("eid_role_map_file", eid_role_map)


# %%
#prepare data frame for email sender reciever graph
send_rec_dict={}
email_raw_df = email_raw_df['2010-01-01':'2010-06-1']
for index, row in email_raw_df.iterrows():
    # print (row['from'], row['to'])
    # print()
    if row['from'] not in email_eid_map.keys():
        send_eid_key = 'OUT101'
    else:
        send_eid_key= email_eid_map[row['from']]
    
    if send_eid_key not in send_rec_dict.keys():
        send_rec_dict[send_eid_key]=[]

    rec_list = row['to'].split(";") + str(row['cc']).split(";") + str(row['bcc']).split(";")
    rec_list = [x for x in rec_list if str(x) != 'nan']
    # print (rec_list)
    rec_list = [email_eid_map[x] if x in email_eid_map.keys() else 'OUT101' for x in rec_list  ]
    # print (rec_list)

    send_rec_dict[send_eid_key]+=rec_list

pickle_store("send_rec_dict_file", send_rec_dict)


# %%
#create a graph from dictionary: send_rec_dict
def create_nxDiGraph():
    send_rec_dict=pickle_restore("send_rec_dict_file")
    def get_wt_edge_list(set_list, edge_list):
        res_triple=[]
        for x in set_list:
            i=0
            for y in edge_list:
                if y == x:
                    i+=1
            triple = (x[0], x[1], i)
            res_triple.append(triple)
        # print (res_triple)
        return res_triple

    G = nx.DiGraph()
    G.add_nodes_from(send_rec_dict.keys())
    # send_rec_dict['LAP0338']
    for key in send_rec_dict.keys():
        edge_list = [ (key, j) for j in send_rec_dict[key] ]
        set_list = list(set(edge_list))
        weighted_edge_list = get_wt_edge_list(set_list, edge_list)
        
        # print(weighted_edge_list)
        G.add_weighted_edges_from(weighted_edge_list) 

    # draw(G)
    #########################################
    # Created a graph. Now cross check it by drawing as below
    #########################################

    #if need to see smaller grqaph use this
    # lst = list(send_rec_dict.keys())
    # lst
    # G = nx.subgraph(G, lst[0:40])

    # pos=nx.spring_layout(G)
    # nx.draw_networkx(G,pos)
    # labels = nx.get_edge_attributes(G,'weight')
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    # nx.draw(G, with_labels=True)
    return G


# %%
#using graph create communities

import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

# load the karate club graph
G = create_nxDiGraph()
G = nx.Graph(G)  #louvian takes undirected graph, convert to undirected
#first compute the best partition
partition = community_louvain.best_partition(G, weight='weight')

# compute the best partition
partition = community_louvain.best_partition(G, weight='weight')

# draw the community graph
pos = nx.spring_layout(G)#nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)

labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels, font_size=4)

# plt.show() #uncomment to view plot

pickle_store("community_louvian_file", partition)


