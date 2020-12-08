from search import *


def main():
    maze, start, end = read_maze('MazeData.txt')

    res, visited, visited_list = UCS(maze, start, end)
    plt.figure()
    plt.title('UCS')
    draw_map(getCoord(maze, ['0', 'S', 'E']), plt.scatter, 'b', '.')
    draw_map(getCoord(visited, [1]), plt.scatter, 'y', '*')
    draw_map(start, plt.scatter, 'orange', 'x')
    draw_map(end, plt.scatter, 'r', 'x')
    draw_map(([obj[0] for obj in res], [obj[1]
                                        for obj in res]), plt.plot, 'g', '.')
    plt.savefig('img/UCS.svg')
    plt.show()

    for method in heuristic_methods:
        res, visited, visited_list = A_star(
            maze, start, end, heuristic=heuristic_methods[method])
        plt.figure()
        plt.title(method)
        draw_map(getCoord(maze, ['0', 'S', 'E']), plt.scatter, 'b', '.')
        draw_map(getCoord(visited, [1]), plt.scatter, 'y', '*')
        draw_map(start, plt.scatter, 'orange', 'x')
        draw_map(end, plt.scatter, 'r', 'x')
        draw_map(([obj[0] for obj in res], [obj[1]
                                            for obj in res]), plt.plot, 'g', '.')
        plt.savefig('img/{}.svg'.format(method))
        plt.show()


if __name__ == "__main__":
    main()
