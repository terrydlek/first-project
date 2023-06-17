def solution(m, musicinfos):
    answer = ''
    re = []
    for i in musicinfos:
        st, ed, name, fos = i.split(",")
        sth, stm = st.split(":")
        edh, edm = ed.split(":")
        time = (int(edh) * 60 + int(edm)) - (int(sth) * 60 + int(stm))
        moc, remnant = time // (len(fos) - fos.count("#")), time % (len(fos) - fos.count("#"))
        fos_sharp = []
        for i in range(len(fos)):
            if i == len(fos) - 1:
                if fos[i].isalpha():
                    fos_sharp.append([fos[i], 0])
            elif fos[i].isalpha():
                if fos[i + 1] == "#":
                    fos_sharp.append([fos[i], "#"])
                else:
                    fos_sharp.append([fos[i], 0])
        song = fos_sharp * moc + fos_sharp[:remnant]
        foss = ""
        for i in fos_sharp:
            if i[1] == "#":
                foss += i[0] + "#"
            else:
                foss += i[0]
        sing = ""
        for i in song:
            if i[1] == "#":
                sing += i[0] + "#"
            else:
                sing += i[0]
        foss_len = len(m)
        print(sing, m)
        for i in range(len(sing)):
            if sing[i:i + foss_len] == m:
                if (i + foss_len < len(sing) and sing[i + foss_len] != "#") or i + foss_len == len(sing):
                    #print("sing:", i, sing[i + foss_len])
                    re.append([name, time])
                    break
    re = sorted(re, key=lambda x: x[1], reverse=True)
    if not re:
        return "(None)"
    else:
        answer = re[0][0]
    return answer


#print(solution("ABCDEFG", ["12:00,12:14,HELLO,CDEFGAB", "13:00,13:05,WORLD,ABCDEF"]))
#print(solution("CC#BCC#BCC#BCC#B", ["03:00,03:30,FOO,CC#B", "04:00,04:08,BAR,CC#BCC#BCC#B"]))
#print(solution("ABC", ["12:00,12:14,HELLO,C#DEFGAB", "13:00,13:05,WORLD,ABCDEF"]))
print(solution("CC#BCC#BCC#", ["03:00,03:08,FOO,CC#B"]))
