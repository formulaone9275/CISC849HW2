import matplotlib.pyplot as plt

loss_dep=[583.8622325211763, 404.0089201852679, 335.1364599764347, 279.875940695405, 242.30250096321106, 222.98737635463476, 212.05199632048607, 191.54173134453595, 170.74025401799008, 156.9942898368463, 147.78290724311955, 144.17831039405428, 130.16488275979646, 131.0473432210274, 123.03906101925531, 117.87332376418635, 115.41668877081247, 110.97502851829631, 109.13174777751556, 106.74936167674605, 104.2687337291427, 97.4099851307401, 94.7528698722308, 95.29954234547768, 90.92932648732676, 89.3285802610917, 89.49860663837171, 85.83928351989016, 83.91465458942184, 83.113215344536, 80.29406848634244, 80.64083873290474, 80.46355240589764, 79.30052923790936, 79.65573342950665, 78.16998140143187, 79.8614749979497, 77.6959975339123, 74.34611884471633, 75.89855232925038, 73.88901670577616, 72.95839928055102, 74.28837108245352, 72.29272790747928, 73.30465763628581, 71.5849595413656, 70.63728434169025, 69.79083473977516, 69.08548720557155, 70.46976208275919]
loss_rgb=[580.8195770829916, 228.13061121013016, 124.09329845150933, 82.45311790858977, 54.43029253766872, 46.87837376806419, 32.80545838224862, 28.133024342085662, 23.743886660940916, 25.30797617981898, 12.961781796624848, 10.282835193024638, 11.941844346170317, 10.792756714333855, 7.069988507702234, 4.20896682044183, 6.795728074909652, 5.892327427735803, 4.258413895815415, 4.881247176597524, 3.9091255886558374, 4.245030941413161, 1.9264557055690208, 3.047714982613937, 3.051691771931699, 1.236401936767547, 1.4179072468350702, 1.3735261876064433, 1.4433358109406917, 0.7234347985406036, 0.7977383006112921, 1.7604433892307156, 0.8696919102596485, 1.325474544001625, 1.9134471275154112, 0.8960283894025096, 0.8062404769581324, 0.9640333062918525, 1.1622863554663914, 0.637680267321727, 0.7658467253639847, 0.4750549865927578, 0.5368386091607511, 1.1209477951253648, 0.5077711750623202, 0.31585270012212385, 0.7505423612691526, 0.4223687880348539, 0.42366407381783144, 0.2713863825210989]

plt.figure()
x_axle=[(a+1) for a in range(len(loss_rgb))]
plt.plot(x_axle, loss_rgb,linewidth=2)
plt.title('Loss change of RGB images ', fontsize=20)
plt.xlabel('Epoch Time', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.show()


plt.figure()
x_axle=[(a+1) for a in range(len(loss_dep))]
plt.plot(x_axle, loss_dep,linewidth=2)
plt.title('Loss change of Depth images ', fontsize=20)
plt.xlabel('Epoch Time', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.show()