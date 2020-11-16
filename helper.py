import pickle

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

def get_anom_list(events_dic):
    anom_user_lst=[]
    for anom_usr in events_dic:
        a_ev = events_dic[anom_usr]
        # print('dict a_ev=', a_ev['anomalies_dict'])
        for k, v in a_ev['anomalies_dict'].items():
            if v != 0:
                anom_user_lst.append(anom_usr)
                # break --> avoids reporting multiple times same user. comment to see if I can remove naive users from flase positive cases.
    return anom_user_lst
