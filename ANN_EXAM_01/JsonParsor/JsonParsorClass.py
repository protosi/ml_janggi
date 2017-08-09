'''
Created on 2017. 8. 2.

@author: 3F8VJ32
'''
import pymysql
import numpy as np
import copy
import json
class JsonParsorClass(object):
    
    sqlConstant = "SELECT * FROM tb_janggi_constant";
    sqlJanggiGame = "SELECT * FROM tb_janggi_game where jg_winner != '' order by rand() limit 0, 1000"
    sqlJanggiPan ="SELECT * FROM tb_janggi_pan where jg_idx = %s"
    sqlRandomPan ="select * from tb_janggi_pan as A inner join tb_janggi_game as B using (jg_idx) where jp_idx >= {} and A.jp_turn_count > B.jg_max_turn_count * {} and jp_winner != '' and jp_turn_flag != ''  limit 0, 1"
    
    sqlMaxPan = "select max(jp_idx) from tb_janggi_pan"
    
    def __init__(self, host='localhost', port=3306, user='ml_user', passwd='mleoqkr!@', db='tensorflow', charset='utf8'):
        self.constantDic = {};
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db
        self.charset=charset
        self.initConstantDic()
    
    def initConstantDic(self):
        cur = self._selectQuery(self.sqlConstant);
        dic = {};
        for row in cur:
            dic.__setitem__(row[0], row[1])
        self.constantDic = dic;
    
    def getGameList(self):
        cur = self._selectQuery(self.sqlJanggiGame)
        rt = []
        
        for row in cur:
            rt.append({"idx": row[0], "title":row[1], "win": row[4], "turn": row[5]})
        return rt
    
    
    
    
    def getRandomPanList(self, list_size = 10, percent = 0.0):
    
        cur = self._selectQuery(self.sqlMaxPan)
        max = 0;     
        for row in cur:
            max = row[0]
        print ("extract start!")
        rt = []
        for i in range(list_size):
            cur =  self._selectQuery(self.sqlRandomPan.format(np.floor(np.random.rand()* max) , percent))
            for row in cur:
                rt.append(self.changeRowToDic(row))
                
        print ("extract done!")
        return rt;
        
        
    def changeRowToDic(self, row): 
        state = self._parseJson(json.loads(row[2]), row[10])
        if row[10] == 'HAN':
            next_state = self._parseJson(json.loads(row[3]), 'CHO')
        elif row[10] == 'CHO':
            next_state = self._parseJson(json.loads(row[3]), 'HAN')
        else: 
            return None
        win = self.constantDic.get(row[4])
        turnCount = row[5]
        pre_x = row[6]
        pre_y = row[7]
        new_x = row[8]
        new_y = row[9]
        turnFlag = self.constantDic.get(row[10])
        moveUnit = row[11]
        done = row[12]
        
        return {"win":win, "turnCount":turnCount, "state": state, 
                "next_state": next_state, "pre_x": pre_x, "pre_y" : pre_y, 
                "new_x": new_x, "new_y": new_y, "turnFlag": turnFlag, 
                "moveUnit": moveUnit, "done": done}
    
    def getPanList(self, game_idx):
        cur = self._selectQuery(self.sqlJanggiPan, game_idx)
        rt = []
        for row in cur:
            rt.append(self.changeRowToDic(row))
        return rt
    def _getConn(self):

        return pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db, charset=self.charset)
    
    def _selectQuery(self, sql, opt = None):
        conn = self._getConn();
        cur = conn.cursor();
        if opt == None:
            cur.execute(sql)
        else: 
            cur.execute(sql, opt)
        cur.close();
        conn.close();
        return cur;
        
    def _parseJson(self, JsonParsor, turnFlag):
        rt = np.zeros((10, 9, 3))
        for i in range(len(JsonParsor)):
            for j in range(len(JsonParsor[i])):
                if JsonParsor[i][j] == 0:
                    continue;
                # HAN
                if "R_" in JsonParsor[i][j]:
                    rt[i][j][1] = self.constantDic.get(JsonParsor[i][j])
                    if(turnFlag == 'HAN'):
                        rt[i][j][2] = 1
                elif "B_" in JsonParsor[i][j]:
                    rt[i][j][0] = self.constantDic.get(JsonParsor[i][j])
                    if(turnFlag == 'CHO'):
                        rt[i][j][2] = 1
        return rt;
        
        