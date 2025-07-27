import "dotenv/config";

import {
    DevelopmentTeamManager,
    DevelopmentTeamManagerInvocationTask,
} from "./teams/development/team-manager";

(async () => {
    const teamManager = await new DevelopmentTeamManager().init();
    console.log("Team manager initialized");

    console.log("Invoking add feature task");
    await teamManager
        .invoke({
            task: DevelopmentTeamManagerInvocationTask.AddFeature,
            data: {
                description:
                    "Add more useful git tools to src/tools/git.ts such as opening a pull request, diffing changes and getting git status.",
            },
        })
        .catch((error) => {
            console.error("Error invoking add feature task:", error);
        });
    console.log("Add feature task complete");

    await teamManager.dispose();
})();
