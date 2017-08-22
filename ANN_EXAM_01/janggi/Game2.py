'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''
import copy
import time
from Unit import Unit
from UnitCha import UnitCha
from UnitGung import UnitGung
from UnitJol import UnitJol
from UnitMa import UnitMa
from UnitPo import UnitPo
from UnitSa import UnitSa
from UnitSang import UnitSang
import random
import numpy as np
from time import sleep
class Game():
    
    # 1 = 초, 2 = 한
    turn = 1
    turnCount = 1
    # 초, 한의 스코어
    choScore = 0
    hanScore = 0
    
    isGame = False
    
    # minmax알고리즘에서 내다볼 수의 수
    m_depth = 2
    
    '''
    좌표계 - 예시
    Y   X      0       1       2       3       4       5       6       7       8
    0    [ B_CHA,   B_MA, B_SANG,   B_SA,      0,   B_SA, B_SANG,   B_MA,  B_CHA],
    1    [     0,      0,      0,      0, B_GUNG,      0,      0,      0,      0],
    2    [     0,   B_PO,      0,      0,      0,      0,      0,   B_PO,      0],
    3    [ B_JOL,      0,  B_JOL,      0,  B_JOL,      0,  B_JOL,      0,  B_JOL],
    4    [     0,      0,      0,      0,      0,      0,      0,      0,      0], 
    5    [     0,      0,      0,      0,      0,      0,      0,      0,      0], 
    6    [ R_JOL,      0,  R_JOL,      0,  R_JOL,      0,  R_JOL,      0,  R_JOL], 
    7    [     0,      0,   R_PO,      0,      0,      0,      0,   R_PO,      0], 
    8    [     0,      0,      0,      0, R_GUNG,      0,      0,      0,      0], 
    9    [ R_CHA,   R_MA, R_SANG,   R_SA,      0,   R_SA, R_SANG,   R_MA,  R_CHA]
    '''
    map = []
    #unitList = [];
    
    

    def __init__(self):
        self.initGame()
        
    def initGame(self):
        self.initMap();
        
        # 턴 초기화
        self.turn = 1
        self.turnCount = 1
        # 스코어 초기화
        self.choScore = 0
        self.hanScore = 0
        self.isGame =True
        
        # 1. CHO
            #CHA
        self.addObjToMap(0, 0, UnitCha(1, self));
        self.addObjToMap(8, 0, UnitCha(1, self));
        
            #MA
        self.addObjToMap(1, 0, UnitMa(1, self));
        self.addObjToMap(7, 0, UnitMa(1, self));
        
            #SANG
        self.addObjToMap(2, 0, UnitSang(1, self));
        self.addObjToMap(6, 0, UnitSang(1, self));
        
            #SA
        self.addObjToMap(3, 0, UnitSa(1, self));
        self.addObjToMap(5, 0, UnitSa(1, self));
        
            #GUNG
        self.addObjToMap(4, 1, UnitGung(1, self));
        
            #PO
        self.addObjToMap(1, 2, UnitPo(1, self));
        self.addObjToMap(7, 2, UnitPo(1, self));
        
            #JOL
        self.addObjToMap(0, 3, UnitJol(1, self));
        self.addObjToMap(2, 3, UnitJol(1, self));
        self.addObjToMap(4, 3, UnitJol(1, self));
        self.addObjToMap(6, 3, UnitJol(1, self));
        self.addObjToMap(8, 3, UnitJol(1, self));
        
        
        # 2. HAN
            #CHA
        self.addObjToMap(0, 9, UnitCha(2, self));
        self.addObjToMap(8, 9, UnitCha(2, self));
        
            #MA        
        self.addObjToMap(1, 9, UnitMa(2, self));
        self.addObjToMap(7, 9, UnitMa(2, self));
        
            #SANG        
        self.addObjToMap(2, 9, UnitSang(2, self));
        self.addObjToMap(6, 9, UnitSang(2, self));
        
            #SA        
        self.addObjToMap(3, 9, UnitSa(2, self));
        self.addObjToMap(5, 9, UnitSa(2, self));
        
            #GUNG
        self.addObjToMap(4, 8, UnitGung(2, self));
        
            #PO
        self.addObjToMap(1, 7, UnitPo(2, self));
        self.addObjToMap(7, 7, UnitPo(2, self));
        
            #JOL
        self.addObjToMap(0, 6, UnitJol(2, self));
        self.addObjToMap(2, 6, UnitJol(2, self));
        self.addObjToMap(4, 6, UnitJol(2, self));
        self.addObjToMap(6, 6, UnitJol(2, self));
        self.addObjToMap(8, 6, UnitJol(2, self));
    
    
    
    '''
        state를 map 배치로 전환하는 함수
    '''
    def setMapByState(self, state):
        
        # 게임 및 맵 초기화
        self.initGame()
        self.initMap()
        
        self.turn = -1
        chomap = state[0]
        hanmap = state[1]
        turnmap = state[2]
        
        # map 처리
        for y in range(len(chomap)):
            for x in range(len(chomap[y])):
                
                if self.turn == -1 and turnmap[y][x] == 1 and chomap[y][x] != 0:
                    self.setTurn(1)
                elif self.turn == -1 and turnmap[y][x] == 1 and hanmap[y][x] != 0:
                    self.setTurn(2)
            
                # 졸 처리
                if chomap[y][x] == UnitJol.staticScore:
                    self.addObjToMap(x, y, UnitJol(1, self));
                
                if hanmap[y][x] == UnitJol.staticScore:
                    self.addObjToMap(x, y, UnitJol(2, self));
                    
                # 상 처리
                if chomap[y][x] == UnitSang.staticScore:
                    self.addObjToMap(x, y, UnitSang(1, self));
                
                if hanmap[y][x] == UnitSang.staticScore:
                    self.addObjToMap(x, y, UnitSang(2, self));
                    
                # 마 처리
                if chomap[y][x] == UnitMa.staticScore:
                    self.addObjToMap(x, y, UnitMa(1, self));
                
                if hanmap[y][x] == UnitMa.staticScore:
                    self.addObjToMap(x, y, UnitMa(2, self));
                    
                # 포 처리
                if chomap[y][x] == UnitPo.staticScore:
                    self.addObjToMap(x, y, UnitPo(1, self));
                
                if hanmap[y][x] == UnitPo.staticScore:
                    self.addObjToMap(x, y, UnitPo(2, self));
                    
                # 차 처리
                if chomap[y][x] == UnitCha.staticScore:
                    self.addObjToMap(x, y, UnitCha(1, self));
                
                if hanmap[y][x] == UnitCha.staticScore:
                    self.addObjToMap(x, y, UnitCha(2, self));
                    
                # 사 처리
                if chomap[y][x] == UnitSa.staticScore:
                    self.addObjToMap(x, y, UnitSa(1, self));
                
                if hanmap[y][x] == UnitSa.staticScore:
                    self.addObjToMap(x, y, UnitSa(2, self));
                    
                # 궁 처리
                if chomap[y][x] == UnitGung.staticScore:
                    self.addObjToMap(x, y, UnitGung(1, self));
                
                if hanmap[y][x] == UnitGung.staticScore:
                    self.addObjToMap(x, y, UnitGung(2, self));    
    
    
    def doGame(self, pos):
        
        myTurn  = self.getTurn()
        done = self.setMoveForML(pos)
        enTurn = self.getTurn()
        reward = self.getStageScoreForML(self.getMap(), myTurn) - self.getStageScoreForML(self.getMap(), enTurn)
        
        if(self.turnCount >= 1000):
            done = True
        
        return reward, done
    
    def getPossibleMoveListfromCustomMap(self, _map, flag):
        map = copy.deepcopy(_map)        
        list = np.hstack(map)
        rt = []
        for i in range(len(list)):
            if isinstance(list[i], Unit):
                if(list[i].getFlag() == flag):
                    _, mvlist = list[i].getPossibleMoveList(map)
                    
                    #if len(mvlist) == 0:
                        #self.printCusomMap(map)
                        #print (flag, list[i].getName() + " has no move", list[i].getX(), list[i].getY());

                    for j in range(len(mvlist)):
                        rt.append([list[i].getX(), list[i].getY(), mvlist[j].getXPos(), mvlist[j].getYPos()])
        return rt;
    
    def getPossibleMoveList(self, flag):
        
        map = copy.deepcopy(self.getMap())
        list = np.hstack(map)
        rt = []
        for i in range(len(list)):
            if isinstance(list[i], Unit):
                if(list[i].getFlag() == flag):
                    _, mvlist = list[i].getPossibleMoveList()

                    for j in range(len(mvlist)):
                        rt.append([list[i].getX(), list[i].getY(), mvlist[j].getXPos(), mvlist[j].getYPos()])
        return rt;
    
    def getState(self, move=None):
        
        choMap = self.getUnitMap(1)
        hanMap = self.getUnitMap(2)
        choMap =np.array(choMap).reshape(10, 9, 1)
        hanMap =np.array(hanMap).reshape(10, 9, 1)
        maskMap = np.zeros((10, 9, 1))
        moveMap = np.zeros((10, 9, 1))
        # 초인 경우
        if(self.turn == 1):
            for i in range(len(choMap)):
                for j in range(len(choMap[i])):
                    if choMap[i][j][0] > 0:
                        maskMap[i][j][0] = 1
        elif(self.turn == 2):
            for i in range(len(hanMap)):
                for j in range(len(hanMap[i])):
                    if hanMap[i][j][0] > 0:
                        maskMap[i][j][0] = 1
        
        if move != None:
            pre_x = move[0]
            pre_y = move[1]
            new_x = move[2]
            new_y = move[3]
            
            moveMap[pre_y][pre_x] = -2000
            moveMap[new_y][new_x] = 2000
                    
        rt = np.concatenate((choMap, hanMap, maskMap, moveMap), 2)
        return rt
    
    '''
        state로 CustomMap을 만드는 함수
    '''
    def getCustomMapByState(self, state):
        
        turn = -1
        map = self.getEmptyMap()

        
        # map 처리
        for y in range(len(state)):
            for x in range(len(state[y])):
                
                if turn == -1 and state[y][x][2] == 1 and state[y][x][0] != 0:
                    turn = 1
                elif turn == -1 and state[y][x][2] == 1 and state[y][x][1] != 0:
                    turn = 2
            
                # 졸 처리
                if state[y][x][0] == UnitJol.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitJol(1, self), map);
                
                if state[y][x][1] == UnitJol.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitJol(2, self), map);
                    
                # 상 처리
                if state[y][x][0] == UnitSang.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitSang(1, self), map);
                
                if state[y][x][1] == UnitSang.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitSang(2, self), map);
                    
                # 마 처리
                if state[y][x][0] == UnitMa.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitMa(1, self), map);
                
                if state[y][x][1] == UnitMa.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitMa(2, self), map);
                    
                # 포 처리
                if state[y][x][0] == UnitPo.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitPo(1, self), map);
                
                if state[y][x][1] == UnitPo.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitPo(2, self), map);
                    
                # 차 처리
                if state[y][x][0] == UnitCha.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitCha(1, self), map);
                
                if state[y][x][1] == UnitCha.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitCha(2, self), map);
                    
                # 사 처리
                if state[y][x][0] == UnitSa.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitSa(1, self), map);
                
                if state[y][x][1] == UnitSa.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitSa(2, self), map);
                    
                # 궁 처리
                if state[y][x][0] == UnitGung.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitGung(1, self), map);
                
                if state[y][x][1] == UnitGung.staticScore:
                    map = self.addObjToCustomMap(x, y, UnitGung(2, self), map);    
        
        return map, turn
    
    def getCustomMoveMap(self, _map, pre_x, pre_y, new_x, new_y):
        map = copy.deepcopy(_map)
        map[new_y][new_x] = map[pre_y][pre_x];
        map[pre_y][pre_x] = 0
        return map
    
    def getCustomState(self, _map):
        choMap = self.getCustomUnitMap(1, _map)
        hanMap = self.getCustomUnitMap(2, _map)
        choMap =np.array(choMap).reshape(10, 9, 1)
        hanMap =np.array(hanMap).reshape(10, 9, 1)
        rt = np.concatenate((choMap, hanMap), 2)
        return rt
    
    def getCustomUnitMap(self, flag, _map):
        map = copy.deepcopy(_map)
        
        for i in range(len(map)):
            for j in range(len(map[i])):
                obj = map[i][j];
                if isinstance(obj, Unit):
                    if(obj.getFlag() != flag):
                        map[i][j] = 0
                    else:
                        map[i][j] = obj.getScore()
        return map
    
    def getUnitMap(self, flag):
        map = copy.deepcopy(self.getMap())
        
        for i in range(len(map)):
            for j in range(len(map[i])):
                obj = map[i][j];
                if isinstance(obj, Unit):
                    if(obj.getFlag() != flag):
                        map[i][j] = 0
                    else:
                        map[i][j] = obj.getScore()
        
        return map
    
    def getMinMaxPos(self):
        map = copy.deepcopy(self.getMap())
        return self.doMinMaxForML(map, self.m_depth, None, self.getTurn(), self.getTurn())
    
    def setMoveForML(self, pos):
        
        done = False
        
        pre_x = pos[0]
        pre_y = pos[1]
        new_x = pos[2]
        new_y = pos[3]
        
        # 잘못된 범위 예외처리
        if(pre_y < 0 or  pre_y >= len(self.map)):
            done = True
        
        if(pre_x < 0 or pre_x >= len(self.map[pre_y])):
            done = True
        
        if(new_y < 0 or  new_y >= len(self.map)):
            done = True
        
        if(new_x < 0 or new_x >= len(self.map[new_y])):
            done = True
        
        
        obj = self.map[pre_y][pre_x]
        
        # obj가 Unit의 instance가 아니면 False를 리턴한다.
        if(isinstance(obj, Unit) == False):
            done = True
        
        flag = obj.getFlag()
        
        # 현재 차례와 움직이려는 말이 다른 경우
        if(self.turn != flag):
            done = True
        
        map, _ = obj.getPossibleMoveList()
        
        
        if(map[new_y][new_x] != 1):
            done = True
            
        if done:
            return done;

        target = self.map[new_y][new_x] 
        
        if(isinstance(target, Unit)):
            score = target.getScore()
            # 초인 경우
            if(flag == 1):
                self.choScore += score
            # 한의 경우
            elif(flag == 2):
                self.hanScore += score
        
            if(isinstance(target, UnitGung)):
                self.isGame = False
                done = True
        
        obj.setPos(new_x, new_y)
        self.map[new_y][new_x] = obj
        self.map[pre_y][pre_x] = 0
        
        # turn을 바꾼다
        self.changeTurn()

        return done
    
        
    def doMinMaxForML(self, stage, depth, cut_score, myFlag, turnFlag):
        start_time = time.time()

        # myFlag == turnFlag : 내 차례, max 알고리즘
        # myFlag != turnFlag : 사앧 차례, min 알고리즘
        
        
        depth -= 1
        
        res_score = None
        res_state = None
        
        if turnFlag == 1 :
            oppFlag = 2
        elif turnFlag == 2 :
            oppFlag = 1
            
        if myFlag == 1 :
            enFlag = 2
        elif myFlag == 2 :
            enFlag = 1
        

        node_count = 0
        for row in range(0, len(stage), 1):
            if(row >= len(stage)):
                    break
            
            for col in range(0, len(stage[row]), 1):
                if(row >= len(stage)):
                    break
                if(col >= len(stage[row])):
                    break
                
                # 현재 움직일 유닛
                currUnit = stage[row][col]
                if isinstance(currUnit, Unit):
                    
                    # 현재 움직여야 하는 유닛인 경우에
                    if currUnit.getFlag() == turnFlag:
                        _, poses = currUnit.getPossibleMoveList() 
                        node_count += len(poses)
                        
                        for i in range(0, len(poses)):
                            
                            if poses[i].getYPos() < 0 or poses[i].getYPos() >= len(stage):
                                print("wrong pos : " + str(poses[i].getXPos()) + ", " + str(poses[i].getYPos()))
                                continue
                                
                            if poses[i].getXPos() < 0 or poses[i].getXPos() >= len(stage[poses[i].getYPos()]):
                                print("wrong pos : " + str(poses[i].getXPos()) + ", " + str(poses[i].getYPos()))
                                continue
                            
                            state = [col, row, poses[i].getXPos(), poses[i].getYPos()]
                            # 상대 유닛
                            oppUnit = stage[poses[i].getYPos()][poses[i].getXPos()]
                            
                            # 왕을 잡을 경우, 더 이상 판단할 필요가 없다.
                            if oppUnit != 0 and oppUnit.getFlag() != currUnit.getFlag() and isinstance(oppUnit, UnitGung):
                                
                                # 내 차례인 경우에
                                if turnFlag == myFlag:
                                    # max 카운팅을 하기 위해 양수 return
                                    res_score = oppUnit.getScore()
                                # 남의 차례인 경우에
                                elif turnFlag != myFlag:
                                    # min 카운팅을 하기 위해 음수 return 
                                    res_score = (-1) * oppUnit.getScore()
                                
                                
                                
                                res_state = state
                                
                                # for문 exception을 위한 값 세팅
                                col = 9
                                row = 10
                                break
                            
                            # 왕을 잡지 못했을 경우
                            else:
                                param_stage = self.getChangeStageStatus(stage, col, row, poses[i].getXPos(), poses[i].getYPos())                        
                                state = [col, row, poses[i].getXPos(), poses[i].getYPos()] 
                                
                                if depth == 0:
                                    score = self.getStageScore(param_stage, myFlag) - self.getStageScore(param_stage, enFlag)
                                    
                                    
                                
                                    if res_score == None or (myFlag != turnFlag and res_score > score) or (myFlag == turnFlag and res_score < score) :
                                        res_score = score;
                                        
                                        # 내 차례일 때, max
                                        if(cut_score != None and ((myFlag != turnFlag and res_score <= cut_score) or myFlag == turnFlag and res_score >= cut_score)) :
                                            col = 9
                                            row = 10
                                            break
                                    continue
                                
                                
                                score = self.doMinMaxForML(param_stage, depth, res_score, myFlag, oppFlag) 
                                

                                if res_score == None:
                                    res_score = score
                                    res_state = state
                                    if(cut_score != None and ((myFlag == turnFlag and res_score > cut_score) or (myFlag != turnFlag and score < cut_score))):
                                        col = 9
                                        row = 10
                                        break
                                elif (myFlag == turnFlag and score > res_score) or (myFlag != turnFlag and score < res_score) :
                                    res_score = score
                                    res_state = state
                                    if(cut_score != None and ((myFlag == turnFlag and res_score > cut_score) or (myFlag != turnFlag and score < cut_score))):
                                        col = 9
                                        row = 10
                                        break
                                    
        # for 문의 종료                            
        if depth < self.m_depth -1:
            return res_score
        return res_state
    
    def doMinMax(self, stage, depth, cut_score, myFlag, turnFlag):
        start_time = time.time()

        # myFlag == turnFlag : 내 차례, max 알고리즘
        # myFlag != turnFlag : 사앧 차례, min 알고리즘
        
        
        depth -= 1
        
        res_score = None
        res_state = None
        
        if turnFlag == 1 :
            oppFlag = 2
        elif turnFlag == 2 :
            oppFlag = 1
            
        if myFlag == 1 :
            enFlag = 2
        elif myFlag == 2 :
            enFlag = 1
        

        node_count = 0
        for row in range(0, len(stage), 1):
            if(row >= len(stage)):
                    break
            
            for col in range(0, len(stage[row]), 1):
                if(row >= len(stage)):
                    break
                if(col >= len(stage[row])):
                    break
                
                # 현재 움직일 유닛
                currUnit = stage[row][col]
                if isinstance(currUnit, Unit):
                    
                    # 현재 움직여야 하는 유닛인 경우에
                    if currUnit.getFlag() == turnFlag:
                        _, poses = currUnit.getPossibleMoveList() 
                        node_count += len(poses)
                        
                        for i in range(0, len(poses)):
                            
                            if poses[i].getYPos() < 0 or poses[i].getYPos() >= len(stage):
                                print("wrong pos : " + str(poses[i].getXPos()) + ", " + str(poses[i].getYPos()))
                                continue
                                
                            if poses[i].getXPos() < 0 or poses[i].getXPos() >= len(stage[poses[i].getYPos()]):
                                print("wrong pos : " + str(poses[i].getXPos()) + ", " + str(poses[i].getYPos()))
                                continue
                            
                            state = [col, row, poses[i].getXPos(), poses[i].getYPos()]
                            # 상대 유닛
                            oppUnit = stage[poses[i].getYPos()][poses[i].getXPos()]
                            
                            # 왕을 잡을 경우, 더 이상 판단할 필요가 없다.
                            if oppUnit != 0 and oppUnit.getFlag() != currUnit.getFlag() and isinstance(oppUnit, UnitGung):
                                
                                # 내 차례인 경우에
                                if turnFlag == myFlag:
                                    # max 카운팅을 하기 위해 양수 return
                                    res_score = oppUnit.getScore()
                                # 남의 차례인 경우에
                                elif turnFlag != myFlag:
                                    # min 카운팅을 하기 위해 음수 return 
                                    res_score = (-1) * oppUnit.getScore()
                                
                                
                                
                                res_state = state
                                
                                # for문 exception을 위한 값 세팅
                                col = 9
                                row = 10
                                break
                            
                            # 왕을 잡지 못했을 경우
                            else:
                                param_stage = self.getChangeStageStatus(stage, col, row, poses[i].getXPos(), poses[i].getYPos())                        
                                state = [col, row, poses[i].getXPos(), poses[i].getYPos()] 
                                
                                if depth == 0:
                                    score = self.getStageScore(param_stage, myFlag) - self.getStageScore(param_stage, enFlag)
                                    
                                    
                                
                                    if res_score == None or (myFlag != turnFlag and res_score > score) or (myFlag == turnFlag and res_score < score) :
                                        res_score = score;
                                        
                                        # 내 차례일 때, max
                                        if(cut_score != None and ((myFlag != turnFlag and res_score <= cut_score) or myFlag == turnFlag and res_score >= cut_score)) :
                                            col = 9
                                            row = 10
                                            break
                                    continue
                                
                                
                                score = self.doMinMax(param_stage, depth, res_score, myFlag, oppFlag) 
                                

                                if res_score == None:
                                    res_score = score
                                    res_state = state
                                    if(cut_score != None and ((myFlag == turnFlag and res_score > cut_score) or (myFlag != turnFlag and score < cut_score))):
                                        col = 9
                                        row = 10
                                        break
                                elif (myFlag == turnFlag and score > res_score) or (myFlag != turnFlag and score < res_score) :
                                    res_score = score
                                    res_state = state
                                    if(cut_score != None and ((myFlag == turnFlag and res_score > cut_score) or (myFlag != turnFlag and score < cut_score))):
                                        col = 9
                                        row = 10
                                        break
                                    
        # for 문의 종료                            
        
                                
        # 이게 뭔 시츄레이션이지?
        
        

        if depth < self.m_depth -1:
            return res_score

        
        # 여기에 이동을 처리한다. (왜?)
        
        # your code
        elapsed_time = time.time() - start_time
        print("elapsed_time : " + str(elapsed_time))
        self.setMove(res_state[0], res_state[1], res_state[2], res_state[3])
    
    def getStageScoreForML(self, stage, flag):
        
        score = 0 
        for row in range(0, len(stage)):
            for col in range(0, len(stage[row])):
                
                if stage[row][col] != 0 and stage[row][col].getFlag() == flag :
                    score += stage[row][col].getScore()
                    
        #score += random.randrange(1, 100) - random.randrange(1, 100)
        return score
    
    
    def getStageScore(self, stage, flag):
        
        score = 0 
        for row in range(0, len(stage)):
            for col in range(0, len(stage[row])):
                
                if stage[row][col] != 0 and stage[row][col].getFlag() == flag :
                    score += stage[row][col].getScore()
                    
        score += random.randrange(1, 100) - random.randrange(1, 100)
        return score
    
    def getTurn(self):
        return self.turn
    
    def setTurn(self, turn):
        self.turn = turn
        
    def changeTurn(self):
        self.turnCount += 1
        if self.turn == 1:
            self.turn = 2
        elif self.turn == 2:
            self.turn = 1
        
    def getMove(self, x, y):
        obj = self.map[y][x]
        map = self.getEmptyMap()
        list = []
        # obj가 유닛일 때만 정상적인 길을 보여준다.
        if(isinstance(obj, Unit)):
            map, list = obj.getPossibleMoveList(self.map)
        return map, list
    
    def getScore(self,x, y, flag):
        # 잘못된 범위 예외처리
        target = self.map[x][y]
        
        if(isinstance(target, Unit) == False):
            return 0
        elif target.getFlag() != flag:
            return target.getScore()
        return 0
    
    def getChangeStageStatus(self, stage, f_x, f_y, d_x, d_y):
        
        # return할 stage를 copy 한다
        result = copy.deepcopy(stage)

        result[d_y][d_x] = result[f_y][f_x]
        result[f_y][f_x] = 0
        return result
        
    def setMove(self, pre_x, pre_y, new_x, new_y):
        print(str(pre_x) + ", " + str(pre_y))
        print(str(new_x) + ", " + str(new_y))
        # 잘못된 범위 예외처리
        if(pre_y < 0 or  pre_y >= len(self.map)):
            print("pre_y is wrong value")
            self.isGame = False
            return False
        
        if(pre_x < 0 or pre_x >= len(self.map[pre_y])):
            print("pre_x is wrong value")
            self.isGame = False
            return False
        
        if(new_y < 0 or  new_y >= len(self.map)):
            print("new_y is wrong value")
            self.isGame = False
            return False
        
        if(new_x < 0 or new_x >= len(self.map[new_y])):
            print("new_x is wrong value")
            self.isGame = False
            return False
        
        
        obj = self.map[pre_y][pre_x]
        
        # obj가 Unit의 instance가 아니면 False를 리턴한다.
        if(isinstance(obj, Unit) == False):
            print("map is not Unit")

            return False
        
        flag = obj.getFlag()
        
        # 현재 차례와 움직이려는 말이 다른 경우
        if(self.turn != flag):
            print("wrong flag")
            self.isGame = False
            return False
        
        map, _ = obj.getPossibleMoveList()
        
        print ("===== possible move map =====")
        self.printCusomMap(map)
        print ("=============================")
        
        if(map[new_y][new_x] != 1):
            print("wrong pos")
            self.isGame = False
            return False

        target = self.map[new_y][new_x] 
        
        if(isinstance(target, Unit)):
            score = target.getScore()
            print("score is %s" % score)
            # 초인 경우
            if(flag == 1):
                self.choScore += score
            # 한의 경우
            elif(flag == 2):
                self.hanScore += score
        
            if(isinstance(target, UnitGung)):
                print("왕 잡힘")
                self.isGame = False
                self.printMap();
        
        obj.setPos(new_x, new_y)
        self.map[new_y][new_x] = obj
        self.map[pre_y][pre_x] = 0
        
        # turn을 바꾼다
        self.changeTurn()

        return True
    
    def printMap(self):
        print ("===== current move map =====")
        print("current Turn: " + str(self.turnCount))
        print("초 스코어: " + str(self.choScore))
        print("한 스코어: " + str(self.hanScore))
        for i in range(0, len(self.map)):
            if(i == 0):
                print("%6s" % "Y\X", end='')
                for k in range(0, len(self.map[i])):
                    print("%6s" % k, end='')
                print('')
            for j in range(0, len(self.map[i])):
                if(j == 0):
                    print("%6s" % i, end='')
                print("%6s" % self.map[i][j], end='')
            print('')
        print ("============================")
    
    def printCusomMap(self, map):
        for i in range(0, len(map)):
            if(i == 0):
                print("%6s" % "Y\X", end='')
                for k in range(0, len(map[i])):
                    print("%6s" % k, end='')
                print('')
            for j in range(0, len(map[i])):
                if(j == 0):
                    print("%6s" % i, end='')
                print("%6s" % map[i][j], end='')
            print('')
    
    def getMap(self):
        return self.map;
    
    def setMap(self, map):
        self.map = map;
        
    def addObjToCustomMap(self, x, y, obj, map):
        map[y][x] = obj
        if isinstance(obj, Unit) == True:
            obj.setPos(x, y);
        return map
    
    def addObjToMap(self, x, y, obj):
        self.map[y][x] = obj;    
        if isinstance(obj, Unit) == True:
            obj.setPos(x, y);

    
    def getEmptyMap(self):
        #y축 부터
        rt = []
        for i in range(0, 10):
            temp = [] 
            for j in range(0, 9):
                temp.append(0)
            rt.append(temp)
        return rt
        
    def initMap(self):
        #y축 부터
        self.map = []
        for i in range(0, 10):
            temp = [] 
            for j in range(0, 9):
                temp.append(0)
            self.map.append(temp)