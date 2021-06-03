CREATE CONSTRAINT ON (main:Main_table) ASSERT main.INDEX_ID IS UNIQUE;

USING PERIODIC COMMIT 1000
LOAD CSV WITH HEADERS FROM 'file:///data/test_embedding_100.csv' AS row
CREATE (main:Main_table {PAIR_ID:                    row.pair_id,
                         INDEX_ID:                   toInteger(row.main_id),
                         TARGET:                     null,
                         USER_ID:                    row.user_id,
                         SKU_ID:                     row.sku_id,
                         TIME:                       toInteger(row.time),
                         FZ_0:                       toFloat(row.fz_0),
                         FZ_1:                       row.fz_1,
                         FZ_2:                       toFloat(row.fz_2),
                         FZ_3:                       toFloat(row.fz_3),
                         FZ_4:                       toFloat(row.fz_4),
                         FZ_5:                       row.fz_5,
                         FZ_6:                       row.fz_6,
                         FZ_7:                       toFloat(row.fz_7),
                         FZ_8:                       row.fz_8,
                         FZ_9:                       row.fz_9,
                         FZ_10:                       toFloat(row.fz_10),
                         FZ_11:                       toFloat(row.fz_11),
                         FZ_12:                       toFloat(row.fz_12),
                         FZ_13:                       toFloat(row.fz_13),
                         FZ_14:                       row.fz_14,
                         FZ_15:                       toFloat(row.fz_15),
                         FZ_16:                       toFloat(row.fz_16),
                         FZ_17:                       row.fz_17,
                         FZ_18:                       toFloat(row.fz_18),
                         FZ_19:                       toFloat(row.fz_19),
                         FZ_20:                       toFloat(row.fz_20),
                         FZ_21:                       toFloat(row.fz_21),
                         FZ_22:                       toFloat(row.fz_22),
                         FZ_23:                       row.fz_23,
                         FZ_24:                       toFloat(row.fz_24),
                         FZ_25:                       row.fz_25,
                         FZ_26:                       toFloat(row.fz_26),
                         FZ_27:                       toFloat(row.fz_27),
                         FZ_28:                       toFloat(row.fz_28),
                         FZ_29:                       toFloat(row.fz_29),
                         FZ_30:                       toFloat(row.fz_30),
                         FZ_31:                       toFloat(row.fz_31),
                         FZ_32:                       row.fz_32,
                         FZ_33:                       toFloat(row.fz_33),
                         FZ_34:                       row.fz_34,
                         FZ_35:                       toFloat(row.fz_35),
                         FZ_36:                       toFloat(row.fz_36),
                         FZ_37:                       toFloat(row.fz_37),
                         FZ_38:                       toFloat(row.fz_38),
                         FZ_39:                       toFloat(row.fz_39),
                         FZ_40:                       toFloat(row.fz_40),
                         FZ_41:                       toFloat(row.fz_41),
                         FZ_42:                       toFloat(row.fz_42),
                         FZ_43:                       toFloat(row.fz_43),
                         FZ_44:                       toFloat(row.fz_44),
                         FZ_45:                       toFloat(row.fz_45),
                         FZ_46:                       toFloat(row.fz_46),
                         FZ_47:                       toFloat(row.fz_47),
                         FZ_48:                       toFloat(row.fz_48),
                         FZ_49:                       toFloat(row.fz_49),
                         FZ_50:                       toFloat(row.fz_50),
                         FZ_51:                       row.fz_51,
                         FZ_52:                       row.fz_52,
                         FZ_53:                       toFloat(row.fz_53),
                         FZ_54:                       toFloat(row.fz_54),
                         FZ_55:                       toFloat(row.fz_55),
                         FZ_56:                       toFloat(row.fz_56)
                                                  
});
//WITH main, row
//MATCH (main_old:Main_table {F_CID: row.f_cId})
//USING INDEX main_old:Main_table(F_CID)
//CREATE (main_old)-[:CONTENT_EDGE]->(main)
//WITH main, row
//MATCH (main_old:Main_table {F_UID: row.f_uId})
//USING INDEX main_old:Main_table(F_UID)
//CREATE (main_old)-[:USER_EDGE]->(main);





USING PERIODIC COMMIT 1000
LOAD CSV WITH HEADERS FROM 'file:///data/train_embedding_100.csv' AS row
CREATE (main:Main_table {PAIR_ID:                    row.pair_id,
                         INDEX_ID:                   toInteger(row.main_id),
                         TARGET:                     toBoolean(replace(replace(row.label, '1', 'TRUE'), '0', 'FALSE')),
                         USER_ID:                    row.user_id,
                         SKU_ID:                     row.sku_id,
                         TIME:                       toInteger(row.time),
                         FZ_0:                       toFloat(row.fz_0),
                         FZ_1:                       row.fz_1,
                         FZ_2:                       toFloat(row.fz_2),
                         FZ_3:                       toFloat(row.fz_3),
                         FZ_4:                       toFloat(row.fz_4),
                         FZ_5:                       row.fz_5,
                         FZ_6:                       row.fz_6,
                         FZ_7:                       toFloat(row.fz_7),
                         FZ_8:                       row.fz_8,
                         FZ_9:                       row.fz_9,
                         FZ_10:                       toFloat(row.fz_10),
                         FZ_11:                       toFloat(row.fz_11),
                         FZ_12:                       toFloat(row.fz_12),
                         FZ_13:                       toFloat(row.fz_13),
                         FZ_14:                       row.fz_14,
                         FZ_15:                       toFloat(row.fz_15),
                         FZ_16:                       toFloat(row.fz_16),
                         FZ_17:                       row.fz_17,
                         FZ_18:                       toFloat(row.fz_18),
                         FZ_19:                       toFloat(row.fz_19),
                         FZ_20:                       toFloat(row.fz_20),
                         FZ_21:                       toFloat(row.fz_21),
                         FZ_22:                       toFloat(row.fz_22),
                         FZ_23:                       row.fz_23,
                         FZ_24:                       toFloat(row.fz_24),
                         FZ_25:                       row.fz_25,
                         FZ_26:                       toFloat(row.fz_26),
                         FZ_27:                       toFloat(row.fz_27),
                         FZ_28:                       toFloat(row.fz_28),
                         FZ_29:                       toFloat(row.fz_29),
                         FZ_30:                       toFloat(row.fz_30),
                         FZ_31:                       toFloat(row.fz_31),
                         FZ_32:                       row.fz_32,
                         FZ_33:                       toFloat(row.fz_33),
                         FZ_34:                       row.fz_34,
                         FZ_35:                       toFloat(row.fz_35),
                         FZ_36:                       toFloat(row.fz_36),
                         FZ_37:                       toFloat(row.fz_37),
                         FZ_38:                       toFloat(row.fz_38),
                         FZ_39:                       toFloat(row.fz_39),
                         FZ_40:                       toFloat(row.fz_40),
                         FZ_41:                       toFloat(row.fz_41),
                         FZ_42:                       toFloat(row.fz_42),
                         FZ_43:                       toFloat(row.fz_43),
                         FZ_44:                       toFloat(row.fz_44),
                         FZ_45:                       toFloat(row.fz_45),
                         FZ_46:                       toFloat(row.fz_46),
                         FZ_47:                       toFloat(row.fz_47),
                         FZ_48:                       toFloat(row.fz_48),
                         FZ_49:                       toFloat(row.fz_49),
                         FZ_50:                       toFloat(row.fz_50),
                         FZ_51:                       row.fz_51,
                         FZ_52:                       row.fz_52,
                         FZ_53:                       toFloat(row.fz_53),
                         FZ_54:                       toFloat(row.fz_54),
                         FZ_55:                       toFloat(row.fz_55),
                         FZ_56:                       toFloat(row.fz_56)
                          
});

'''
CALL apoc.periodic.iterate(
"MATCH (a:Main_table) 
with a.F_UID AS id, COLLECT(a) AS mains 
UNWIND mains AS a 
UNWIND mains AS b 
with a,b WHERE ID(a) < ID(b) 
RETURN a,b"
,
"create r = (a)-[:USER_EDGE]->(b);",
{batchSize: 1000});


CALL apoc.periodic.iterate(
"MATCH (a:Main_table) 
with a.F_CID AS id, COLLECT(a) AS mains 
UNWIND mains AS a 
UNWIND mains AS b 
with a,b WHERE ID(a) < ID(b) 
RETURN a,b"
,
"create r = (a)-[:CONTENT_EDGE]->(b);",
{batchSize: 1000});



MATCH (a:Main_table)
WITH a.F_UID AS id, COLLECT(a) AS mains
UNWIND mains AS a
UNWIND mains AS b
WITH a, b
MERGE (a)-[:USER_EDGE]-(b);

MATCH (a:Main_table)
WITH a.F_CID AS id, COLLECT(a) AS mains
UNWIND mains AS a
UNWIND mains AS b
WITH a, b
WHERE ID(a) < ID(b)
MERGE (a)-[:CONTENT_EDGE]->(b);


match (a:Main_table)
with a
match (b:Main_table)
where a.F_CID = b.F_CID
CREATE (a)-[:CONTENT_EDGE]-(b)
'''


//MATCH (a:Main_table),(b:Main_table)
//WHERE a.F_CID = b.F_CID
//CREATE (a)-[:CONTENT_EDGE]->(b);


//MATCH (a:Main_table),(b:Main_table)
//WHERE a.F_UID = b.F_UID
//CREATE (a)-[:USER_EDGE]->(b);










