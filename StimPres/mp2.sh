**********************
****** ECE 5362 ******
** Machine Problem 2 *
*** Shahil Merchant **
******10/25/2024*******

*** fetch cycle ******
 st=0 rt='[pc]-> mar'       imar rac=1 rn=3
 st=1 rt='[[mar]]-> mdr'    read
 st=2 rt='[mdr] -> ir'      omdr iir
 st=3 rt='[pc]+1 -> q'      rac=1 rn=3 ib p1 oadder
 st=4 rt='[q] -> pc'        oq wac=1 wn=3
    cond='ir1512' value=4 nst=70
    cond='ir1512' value=0 nst=20
    cond='ir158' value=0 nst=10 
    nst=0                          

*** halt cycle *****
 st=10 halt

*** single ops ***
 st=20
    cond='ir118' value=4 nst=21     
    nst=10    

*** NEG ********************************************              
 st=21
    cond='ir64' value=0 nst=30     
    cond='ir64' value=1 nst=40      
    cond='ir64' value=2 nst=50     
    cond='ir64' value=3 nst=60     
    nst=10                        

*** direct ***
 st=30 rt='[Rk] -> t2'               it1 rac=3
 st=31 rt='0 - [t1] -> q'            ib comp p1 oadder newc newv
 st=32 rt='[q] -> Rk'                oq wac=3 newz newn
    nst=0

*** indirect ***
 st=40 rt='[Rk] -> mar'              imar rac=3
 st=41 rt='[[mar]] -> mdr'           read
 st=42 rt='[mdr] -> t1'              omdr it1
 st=43 rt='0 - [t1] -> q'            ot1 ib comp p1 oadder newc newv
 st=44 rt='[q] -> mdr'               oq imdr newz newn
 st=45 rt='[mdr] -> [mar]'           omdr write
    nst=0

*** autoincrement ***
 st=50 rt='[Rk] -> mar'              imar rac=3
 st=51 rt='[[mar]] -> mdr'           read
 st=52 rt='[Rk] + 1 -> q'            rac=3 ib p1 oadder
 st=53 rt='[q] -> Rk'                oq wac=3
 st=54 rt='[mdr] -> t1'              omdr it1
 st=55 rt='0 - [t1] -> q'            ot1 ib comp p1 oadder newc newv
 st=56 rt='[q] -> mdr'               oq imdr newz newn
 st=57 rt='[mdr] -> [mar]'           omdr write
    nst=0

*** autodecrement ***
 st=60 rt='[Rk] - 1 -> q'            rac=3 ib comp p1 oadder
 st=61 rt='[q] -> t5'                oq it5
 st=62 rt='0 -[t5] -> q'             ot5 ib comp oadder
 st=63 rt='[q] -> Rk'                oq wac=3
 st=64 rt='[Rk] -> mar'              imar rac=3
 st=65 rt='[[mar]] -> mdr'           read
 st=66 rt='[mdr] -> t1'              omdr it1
 st=67 rt='0 - [t1] -> q'            ot1 ib comp p1 oadder newc newv
 st=68 rt='[q] -> mdr'               oq imdr newz newn
 st=69 rt='[mdr] -> [mar]'           omdr write
    nst=0

                        
*** EXG ********************************************
*** src op ***
 st=70
    cond='ir118' value=0 nst=71     
    cond='ir118' value=1 nst=80   
    cond='ir118' value=2 nst=90    
    cond='ir118' value=3 nst=100   
    nst=10                          

*** direct ***
 st=71 rt='[Rs] -> t1'               it1 rac=2
    nst=110

*** indirect ***
 st=80 rt='[Rs] -> mar'              imar rac=2
 st=81 rt='[[mar]] -> mdr'           read 
 st=82 rt='[mdr] -> t1'              omdr  it1
    nst=110

*** autoincrement  ***
 st=90 rt='[Rs] -> mar'              imar rac=2
 st=91 rt='[[mar]] -> mdr'           read
 st=92 rt='[mdr] -> t1'              omdr it1
 st=93 rt='[Rs] + 1 -> q'            rac=2 ib p1 oadder
 st=94 rt='[q] -> Rs'                oq wac=2
    nst=110

*** autodecrement ***
 st=100 rt='[Rs] - 1 -> q'           rac=2 ib comp p1 oadder
 st=101 rt='[q] -> t5'               oq it5
 st=102 rt='0 - [t5] -> q'           ot5 ib comp oadder
 st=103 rt='[q] -> Rs'               oq wac=2
 st=104 rt='[Rs] -> mar'             imar rac=2
 st=105 rt='[[mar]] -> mdr'          read 
 st=106 rt='[mdr] -> t1'             omdr it1
    nst=110

*** dst op ***
 st=110
    cond='ir64' value=0 nst=111  
    cond='ir64' value=1 nst=120  
    cond='ir64' value=2 nst=130    
    cond='ir64' value=3 nst=140  
    nst=10                        
*** direct ***
 st=111 rt='[Rd] -> t2'              it2 rac=3
    nst=150

*** indirect ***
 st=120 rt='[Rd] -> mar'             imar rac=3
 st=121 rt='[[mar]] -> mdr'          read 
 st=122 rt='[mdr] -> t2'             ib omdr it2
    nst=150

*** autoincrement ***
 st=130 rt='[Rd] -> mar'             imar rac=3
 st=131 rt='[[mar]] -> mdr'          read 
 st=132 rt='[[mdr]] -> t2'           omdr it2
 st=133 rt='[Rd] + 1 -> q'           rac=3 ib p1 oadder
 st=134 rt='[q] -> Rd'               oq wac=3
    nst=150

*** autodecrement ***
 st=140 rt='[Rd] - 1 -> q'           rac=3 ib comp p1 oadder
 st=141 rt='[q] -> t5'               oq it5
 st=142 rt='0 - [t5] -> q'           ot5 ib comp oadder
 st=143 rt='[q] -> Rd'               oq wac=3
 st=144 rt='[Rd] -> mar'             imar rac=3
 st=145 rt='[[mar]] -> mdr'          read 
 st=146 rt='[[mdr]] -> t2'           omdr it2
    nst=150

*** swap ***
 st=150 rt='[t1] -> t3'              it3 ot1
 st=151 rt='[t2] -> t1'              it1 ot2
 st=152 rt='[t3] -> t2'              it2 ot3

*** write src ***
 st=153
    cond='ir118' value=0 nst=154  
    cond='ir118' value=1 nst=160  
    cond='ir118' value=2 nst=160    
    cond='ir118' value=3 nst=160   

*** direct ***
 st=154 rt='[t1] -> Rs'              ot1 wac=2
    nst=190

*** indirect ***
 st=160 rt='[Rs] -> mar'             imar rac=2
 st=161 rt='[t1] -> mdr'             ot1 imdr
 st=162 rt='[mdr] -> [mar]'          omdr write

    nst=190

*** autoincrement  ***
 st=170 rt='[Rs] -> mar'             imar rac=2
 st=171 rt='[t1] -> mdr'             ot1 imdr
 st=172 rt='[mdr] -> [mar]'          omdr write
    nst=190

*** autodecrement ***
 st=180 rt='[Rs] -> mar'             imar rac=2
 st=181 rt='[t1] -> mdr'             ot1 imdr
 st=182 rt='[mdr] -> [mar]'          omdr write
    nst=190

*** write dst ***
 st=190
    cond='ir64' value=0 nst=191     
    cond='ir64' value=1 nst=201     
    cond='ir64' value=2 nst=201    
    cond='ir64' value=3 nst=201     
    nst=10                          

*** direct ***
 st=191 rt='[t2] -> Rd'              ot2 wac=3
    nst=0

*** indirect ***
 st=201 rt='[t2] -> mdr'             ot2 imdr
 st=202 rt='[mdr] -> [mar]'          omdr write
 st=203 rt='[Rd] -> mar'             imar rac=3
    nst=0

*** autoincrement ***
 st=210 rt='[Rd] -> mar'             imar rac=3
 st=211 rt='[t2] -> mdr'             ot2 imdr
 st=212 rt='[mdr] -> [mar]'          omdr write
    nst=0

*** autodecrement ***
 st=220 rt='[Rd] -> mar'             imar rac=3
 st=221 rt='[t2] -> mdr'             ot2 imdr
 st=222 rt='[mdr] -> [mar]'          omdr write
    nst=0
