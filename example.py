from minisbd import SBDetect

text = """
La Révolution française (1789-1799) est une période de bouleversements politiques et sociaux en France et dans ses colonies, ainsi qu'en Europe à la fin du XVIIIe siècle. Traditionnellement, on la fait commencer à l'ouverture des États généraux le 5 mai 1789 et finir au coup d'État de Napoléon Bonaparte le 9 novembre 1799 (18 brumaire de l'an VIII). En ce qui concerne l'histoire de France, elle met fin à l'Ancien Régime, notamment à la monarchie absolue, remplacée par la monarchie constitutionnelle (1789-1792) puis par la Première République.

« Mythe national », la Révolution française a légué de nouvelles formes politiques, notamment au travers de la Déclaration des droits de l'homme et du citoyen de 1789 qui proclame l'égalité des citoyens devant la loi, les libertés fondamentales et la souveraineté de la Nation, se constituant autour d'un État. Elle a entraîné la suppression de la société d'ordres, de la féodalité et des privilèges, une plus grande division de la propriété foncière, la limitation de l'exercice du pouvoir politique, le rééquilibrage des relations entre l'Église et l'État et la redéfinition des structures familiales. Les valeurs et les institutions de la Révolution dominent encore aujourd'hui la vie politique française.
"""

detector = SBDetect("fr", use_gpu=True)
for sent in detector.sentences(text):
    print(f"--> {sent}")